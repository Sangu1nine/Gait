"""
GAIT 감지가 포함된 IMU 센서 데이터 수집 프로그램 (30Hz)
Created: 2025-01-29
Modified: 2025-01-29 - 보행 감지 기능 추가

기능:
- IMU 센서에서 30Hz로 데이터 수집
- Stage1 TensorFlow Lite 모델을 사용한 실시간 보행 감지
- 보행 감지 시에만 WiFi로 데이터 전송
- CSV 파일로 모든 데이터 저장 (보행/비보행 분류 포함)
"""

from smbus2 import SMBus
from bitstring import Bits
import os
import math
import time
import pandas as pd
import datetime
import numpy as np
import socket
import json
import threading
import pickle
from collections import deque
import tensorflow as tf

# 글로벌 변수
bus = SMBus(1)
DEV_ADDR = 0x68

# IMU 레지스터 주소
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0

register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0

# WiFi 통신 설정
WIFI_SERVER_IP = '172.20.10.12'  # 로컬 PC의 IP 주소 (수정됨)
WIFI_SERVER_PORT = 5000  # 통신 포트
wifi_client = None
wifi_connected = False
send_data_queue = []

# 보행 감지 관련 설정
MODEL_PATH = "Gait/models/gait_detection/model.tflite"
SCALER_PATH = "Gait/scalers/gait_detection"  # 스케일러 파일이 있는 디렉토리
WINDOW_SIZE = 60  # Stage1 모델의 윈도우 크기
TARGET_HZ = 30   # 샘플링 레이트
GAIT_THRESHOLD = 0.2  # 보행 감지 임계값 (기본값)

# 보행 감지 전역 변수
interpreter = None
minmax_scaler = None
sensor_buffer = deque(maxlen=WINDOW_SIZE)
gait_detection_enabled = False
last_gait_status = "non_gait"
gait_count = 0
non_gait_count = 0

def read_data(register):
    """IMU 레지스터에서 데이터 읽기"""
    high = bus.read_byte_data(DEV_ADDR, register)
    low = bus.read_byte_data(DEV_ADDR, register+1)
    val = (high << 8) + low
    return val

def twocomplements(val):
    """2의 보수 변환"""
    s = Bits(uint=val, length=16)
    return s.int

def gyro_dps(val):
    """자이로스코프 값을 도/초로 변환"""
    return twocomplements(val)/sensitive_gyro

def accel_ms2(val):
    """가속도 값을 m/s²로 변환"""
    return (twocomplements(val)/sensitive_accel) * 9.80665

def load_gait_detection_model():
    """보행 감지 모델과 스케일러 로딩"""
    global interpreter, minmax_scaler, gait_detection_enabled
    
    try:
        # TensorFlow Lite 모델 로딩
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            print(f"✅ TFLite 모델 로딩 완료: {MODEL_PATH}")
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return False
        
        # MinMax 스케일러 로딩
        scaler_file = os.path.join(SCALER_PATH, "minmax_scaler.pkl")
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                minmax_scaler = pickle.load(f)
            print(f"✅ MinMax 스케일러 로딩 완료: {scaler_file}")
        else:
            print(f"❌ 스케일러 파일을 찾을 수 없습니다: {scaler_file}")
            return False
        
        gait_detection_enabled = True
        print("🚶 보행 감지 시스템 활성화됨")
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 오류: {str(e)}")
        return False

def predict_gait(sensor_data):
    """센서 데이터로 보행 감지 예측"""
    global interpreter, minmax_scaler, last_gait_status
    
    if not gait_detection_enabled or len(sensor_data) != WINDOW_SIZE:
        return "unknown"
    
    try:
        # 데이터 전처리 (stage1_preprocessing.py 참고)
        sensor_array = np.array(sensor_data, dtype=np.float32).reshape(1, WINDOW_SIZE, 6)
        
        # MinMax 스케일링 적용
        n_samples, n_frames, n_features = sensor_array.shape
        sensor_reshaped = sensor_array.reshape(-1, n_features)
        sensor_scaled = minmax_scaler.transform(sensor_reshaped)
        sensor_scaled = sensor_scaled.reshape(n_samples, n_frames, n_features).astype(np.float32)
        
        # TensorFlow Lite 모델 추론
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], sensor_scaled)
        interpreter.invoke()
        
        # 예측 결과 가져오기
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # 확률값을 이진 분류로 변환 (임계값 사용)
        gait_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
        predicted_class = "gait" if gait_probability > GAIT_THRESHOLD else "non_gait"
        
        last_gait_status = predicted_class
        return predicted_class
        
    except Exception as e:
        print(f"⚠️  보행 감지 오류: {str(e)}")
        return "unknown"

def connect_wifi():
    """WiFi 연결 설정"""
    global wifi_client, wifi_connected
    try:
        wifi_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        wifi_client.connect((WIFI_SERVER_IP, WIFI_SERVER_PORT))
        wifi_connected = True
        print(f"✅ WiFi 연결 성공: {WIFI_SERVER_IP}:{WIFI_SERVER_PORT}")
        return True
    except Exception as e:
        print(f"❌ WiFi 연결 실패: {str(e)}")
        wifi_connected = False
        return False

def send_data_thread():
    """데이터 전송 스레드"""
    global send_data_queue, wifi_client, wifi_connected
    
    while wifi_connected:
        if len(send_data_queue) > 0:
            try:
                # 큐에서 데이터 가져오기
                sensor_data = send_data_queue.pop(0)
                # JSON 형식으로 변환하여 전송
                data_json = json.dumps(sensor_data)
                wifi_client.sendall((data_json + '\n').encode('utf-8'))
            except Exception as e:
                print(f"❌ 데이터 전송 오류: {str(e)}")
                wifi_connected = False
                break
        else:
            time.sleep(0.001)

def close_wifi():
    """WiFi 연결 종료"""
    global wifi_client, wifi_connected
    if wifi_client:
        try:
            wifi_client.close()
            print("✅ WiFi 연결 종료")
        except:
            pass
    wifi_connected = False

def main():
    """메인 실행 함수"""
    global sensor_buffer, gait_count, non_gait_count
    
    print("=" * 60)
    print("🚶 GAIT 감지 IMU 센서 데이터 수집 프로그램 (30Hz)")
    print("=" * 60)
    
    # 보행 감지 모델 로딩
    if not load_gait_detection_model():
        print("⚠️  보행 감지 모델을 로딩할 수 없습니다. 모든 데이터를 전송합니다.")
    
    # 센서 초기화
    bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
    
    # 데이터 프레임 준비
    columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'Timestamp', 'GaitStatus']
    data = []
    
    # 파일명 설정 (현재 시간 기반)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gait_imu_data_{timestamp}.csv"
    
    print(f"📄 데이터 저장 파일: {filename}")
    print(f"🎯 보행 감지 임계값: {GAIT_THRESHOLD}")
    print("Ctrl+C를 눌러 수집을 중단하세요")
    
    # WiFi 연결 시도
    wifi_thread = None
    if connect_wifi():
        wifi_thread = threading.Thread(target=send_data_thread)
        wifi_thread.daemon = True
        wifi_thread.start()
    
    # 초기 시간
    start_time = time.time()
    sample_count = 0
    
    try:
        while True:
            # 현재 샘플 시간 계산
            current_time = time.time()
            elapsed = current_time - start_time
            
            # IMU 센서 데이터 읽기
            accel_x = accel_ms2(read_data(register_accel_xout_h))
            accel_y = accel_ms2(read_data(register_accel_yout_h))
            accel_z = accel_ms2(read_data(register_accel_zout_h))
            
            gyro_x = gyro_dps(read_data(register_gyro_xout_h))
            gyro_y = gyro_dps(read_data(register_gyro_yout_h))
            gyro_z = gyro_dps(read_data(register_gyro_zout_h))
            
            # 센서 데이터 (Stage1 preprocessing과 동일한 순서)
            sensor_row = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            
            # 보행 감지를 위한 버퍼에 추가
            sensor_buffer.append(sensor_row)
            
            # 보행 감지 (버퍼가 충분히 찬 경우)
            gait_status = "unknown"
            if len(sensor_buffer) == WINDOW_SIZE:
                gait_status = predict_gait(list(sensor_buffer))
                
                # 통계 업데이트
                if gait_status == "gait":
                    gait_count += 1
                elif gait_status == "non_gait":
                    non_gait_count += 1
                
                # 보행 감지 시에만 WiFi 전송
                if gait_status == "gait" and wifi_connected:
                    sensor_data_wifi = {
                        'timestamp': elapsed,
                        'accel': {'x': accel_x, 'y': -accel_y, 'z': accel_z},  # Y축 반전
                        'gyro': {'x': gyro_x, 'y': -gyro_y, 'z': gyro_z},      # Y축 반전
                        'gait_status': gait_status
                    }
                    send_data_queue.append(sensor_data_wifi)
            
            # 모든 데이터를 CSV에 저장 (보행 상태 포함)
            data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, elapsed, gait_status])
            sample_count += 1
            
            # 30 샘플마다 진행 상황 출력
            if sample_count % 30 == 0:
                total_predictions = gait_count + non_gait_count
                gait_percentage = (gait_count / total_predictions * 100) if total_predictions > 0 else 0
                
                print(f"📊 샘플: {sample_count}, 시간: {elapsed:.2f}s, 샘플링율: {sample_count/elapsed:.2f}Hz")
                print(f"🏃 가속도(m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                print(f"🔄 자이로(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                print(f"🚶 보행 상태: {gait_status} (보행률: {gait_percentage:.1f}%)")
                if wifi_connected:
                    transmitted = "전송됨" if gait_status == "gait" else "전송안됨"
                    print(f"📡 WiFi: 연결됨, 큐길이: {len(send_data_queue)}, 상태: {transmitted}")
                else:
                    print("📡 WiFi: 연결 안됨")
                print("-" * 50)
            
            # 샘플링 레이트 유지 (30Hz)
            next_sample_time = start_time + (sample_count * (1.0 / TARGET_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n🛑 데이터 수집 중단됨!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        
    finally:
        # 데이터프레임 생성 및 저장
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)
        
        # 최종 통계 출력
        total_samples = len(df)
        total_predictions = gait_count + non_gait_count
        gait_percentage = (gait_count / total_predictions * 100) if total_predictions > 0 else 0
        
        print("=" * 60)
        print("📊 수집 완료 통계")
        print("=" * 60)
        print(f"📄 저장 파일: {filename}")
        print(f"📈 총 샘플 수: {total_samples}")
        print(f"🚶 보행 감지: {gait_count}회 ({gait_percentage:.1f}%)")
        print(f"🏃 비보행 감지: {non_gait_count}회 ({100-gait_percentage:.1f}%)")
        print(f"⏱️  총 수집 시간: {elapsed:.2f}초")
        print(f"📊 평균 샘플링율: {total_samples/elapsed:.2f}Hz")
        print("=" * 60)
        
        # 리소스 정리
        close_wifi()
        bus.close()
        print("✅ 모든 리소스 정리 완료")

if __name__ == "__main__":
    main() 