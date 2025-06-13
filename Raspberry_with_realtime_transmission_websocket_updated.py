"""
개선된 라즈베리파이 낙상 감지 시스템
- 간소화된 코드 구조
- 안정적인 데이터 전송
- 낙상 데이터 보장 시스템
"""

import time
import numpy as np
import tensorflow as tf
from collections import deque
import signal
import sys
import pickle
import os
import json
import threading
import asyncio
import websockets
from datetime import datetime, timezone, timedelta
import queue

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    print("SMBus2 라이브러리가 없습니다. 'pip install smbus2' 실행하세요.")
    SENSOR_AVAILABLE = False

# === 설정 ===
DEV_ADDR = 0x68
PWR_MGMT_1 = 0x6B

# 센서 설정
ACCEL_REGISTERS = [0x3B, 0x3D, 0x3F]
GYRO_REGISTERS = [0x43, 0x45, 0x47]
SENSITIVE_ACCEL = 16384.0  # ±2g
SENSITIVE_GYRO = 131.0     # ±250°/s

# 모델 설정
MODEL_PATH = 'models/fall_detection.tflite'
SCALERS_DIR = 'scalers'
SEQ_LENGTH = 150  # 모델 입력 shape와 일치시킴 (1.5초)
STRIDE = 5        # 0.05초마다 예측
SAMPLING_RATE = 100  # 센서 감지/낙상 감지 100Hz 유지
SEND_RATE = 10       # WebSocket 송신 10Hz

# 통신 설정
WEBSOCKET_SERVER_IP = '192.168.0.177'
WEBSOCKET_SERVER_PORT = 8000
USER_ID = "raspberry_pi_01"

# 시간대
KST = timezone(timedelta(hours=9))

class SafeDataSender:
    """안전한 데이터 전송 관리자"""
    def __init__(self):
        self.imu_queue = queue.Queue(maxsize=100)  # 10Hz용으로 크기 축소
        self.fall_queue = queue.Queue(maxsize=100)  # 낙상 데이터는 별도 큐
        self.websocket = None
        self.connected = False
        
    def add_imu_data(self, data):
        """IMU 데이터 추가 (큐가 가득 차면 오래된 것 제거)"""
        try:
            self.imu_queue.put_nowait(data)
        except queue.Full:
            try:
                self.imu_queue.get_nowait()  # 오래된 데이터 제거
                self.imu_queue.put_nowait(data)
            except queue.Empty:
                pass
    
    def add_fall_data(self, data):
        """낙상 데이터 추가 (절대 손실 방지)"""
        try:
            self.fall_queue.put_nowait(data)
            print(f"🚨 낙상 데이터 큐 추가 완료! 대기: {self.fall_queue.qsize()}개")
        except queue.Full:
            print("❌ 낙상 데이터 큐 가득참! 긴급 처리 필요")
    
    async def send_loop(self):
        """데이터 전송 루프"""
        while True:
            try:
                # 낙상 데이터 우선 처리
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data, is_fall=True)
                
                # IMU 데이터 처리 (연결된 경우만)
                elif self.connected and not self.imu_queue.empty():
                    imu_data = self.imu_queue.get_nowait()
                    await self._send_data(imu_data, is_fall=False)
                
                await asyncio.sleep(0.01)  # 10Hz에 맞춰 전송 간격 조정
                
            except Exception as e:
                print(f"전송 루프 오류: {e}")
                await asyncio.sleep(1)
    
    async def _send_data(self, data, is_fall=False):
        """실제 데이터 전송"""
        if not self.websocket:
            if is_fall:
                # 낙상 데이터는 다시 큐에 추가
                self.fall_queue.put_nowait(data)
            return
        
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            await self.websocket.send(json_data)
            
            if is_fall:
                print(f"🚨 낙상 데이터 전송 성공! 신뢰도: {data['data'].get('confidence_score', 0):.2%}")
                
        except Exception as e:
            print(f"데이터 전송 실패: {e}")
            if is_fall:
                # 낙상 데이터는 다시 큐에 추가
                self.fall_queue.put_nowait(data)

class SimpleSensor:
    """간소화된 센서 클래스"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("센서 라이브러리 없음")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        
        # 스케일러 로드
        self.scalers = self._load_scalers()
        print("센서 초기화 완료")
    
    def _load_scalers(self):
        """스케일러 로드"""
        scalers = {}
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        for feature in features:
            try:
                std_path = os.path.join(SCALERS_DIR, f"{feature}_standard_scaler.pkl")
                minmax_path = os.path.join(SCALERS_DIR, f"{feature}_minmax_scaler.pkl")
                
                with open(std_path, 'rb') as f:
                    scalers[f"{feature}_standard"] = pickle.load(f)
                with open(minmax_path, 'rb') as f:
                    scalers[f"{feature}_minmax"] = pickle.load(f)
            except Exception as e:
                print(f"스케일러 로드 실패 {feature}: {e}")
        
        return scalers
    
    def _read_word_2c(self, reg):
        """2의 보수 값 읽기"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val
    
    def get_data(self):
        """센서 데이터 읽기 및 정규화"""
        # 원시 데이터 읽기
        raw_data = []
        for reg in ACCEL_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_ACCEL)
        for reg in GYRO_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_GYRO)
        
        # 정규화
        if self.scalers:
            features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            normalized = []
            for i, feature in enumerate(features):
                val = raw_data[i]
                
                # 표준 스케일링
                if f"{feature}_standard" in self.scalers:
                    scaler = self.scalers[f"{feature}_standard"]
                    val = (val - scaler.mean_[0]) / scaler.scale_[0]
                
                # MinMax 스케일링
                if f"{feature}_minmax" in self.scalers:
                    scaler = self.scalers[f"{feature}_minmax"]
                    val = val * scaler.scale_[0] + scaler.min_[0]
                
                normalized.append(val)
            return np.array(normalized)
        
        return np.array(raw_data)

class SimpleFallDetector:
    """간소화된 낙상 감지기"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("낙상 감지 모델 로드 완료")
    
    def add_data(self, data):
        """데이터 추가"""
        self.buffer.append(data)
        self.counter += 1
    
    def should_predict(self):
        """예측 시점 확인"""
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0
    
    def predict(self):
        """낙상 예측"""
        if len(self.buffer) < SEQ_LENGTH:
            return None
        
        try:
            # 데이터 준비
            data = np.expand_dims(np.array(list(self.buffer)), axis=0).astype(np.float32)
            
            # 예측 실행
            self.interpreter.set_tensor(self.input_details[0]['index'], data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 결과 처리
            fall_prob = float(output.flatten()[0])
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {'prediction': prediction, 'probability': fall_prob}
            
        except Exception as e:
            print(f"예측 오류: {e}")
            return None

def get_current_timestamp():
    """현재 KST 시간"""
    return datetime.now(KST).isoformat()

def create_imu_package(data, user_id):
    """IMU 데이터 패키지 생성"""
    return {
        'type': 'imu_data',
        'data': {
            'user_id': user_id,
            'timestamp': get_current_timestamp(),
            'acc_x': float(data[0]),
            'acc_y': float(data[1]),
            'acc_z': float(data[2]),
            'gyr_x': float(data[3]),
            'gyr_y': float(data[4]),
            'gyr_z': float(data[5])
        }
    }

def create_fall_package(user_id, probability, sensor_data):
    """낙상 데이터 패키지 생성"""
    return {
        'type': 'fall_detection',
        'data': {
            'user_id': user_id,
            'timestamp': get_current_timestamp(),
            'fall_detected': True,
            'confidence_score': float(probability),
            'sensor_data': {
                'acceleration': {'x': float(sensor_data[0]), 'y': float(sensor_data[1]), 'z': float(sensor_data[2])},
                'gyroscope': {'x': float(sensor_data[3]), 'y': float(sensor_data[4]), 'z': float(sensor_data[5])}
            }
        }
    }

async def websocket_handler(data_sender):
    """WebSocket 연결 관리"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    retry_delay = 1
    
    while True:
        try:
            print(f"WebSocket 연결 시도: {url}")
            async with websockets.connect(url) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                retry_delay = 1
                print("✅ WebSocket 연결 성공")
                
                # 데이터 전송 루프 시작
                await data_sender.send_loop()
                
        except Exception as e:
            print(f"WebSocket 연결 실패: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
        
        print(f"재연결 대기: {retry_delay}초")
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)

# IMU 송신 버퍼 (100Hz로 쌓고 10Hz로 송신)
imu_send_buffer = deque(maxlen=SAMPLING_RATE)  # 1초치 버퍼

def main():
    """메인 함수"""
    print("🚀 낙상 감지 시스템 시작")
    print(f"현재 시간 (KST): {get_current_timestamp()}")
    
    # 초기화
    try:
        sensor = SimpleSensor()
        detector = SimpleFallDetector()
        data_sender = SafeDataSender()
    except Exception as e:
        print(f"초기화 실패: {e}")
        return
    
    # 종료 처리
    def signal_handler(sig, frame):
        print("\n프로그램 종료 중...")
        if not data_sender.fall_queue.empty():
            print(f"남은 낙상 데이터: {data_sender.fall_queue.qsize()}개")
            time.sleep(3)
        print("프로그램 종료")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # WebSocket 클라이언트 시작
    def start_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(data_sender))
    
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    
    print("🔄 데이터 수집 시작")
    
    # 초기 버퍼 채우기
    for _ in range(SEQ_LENGTH):
        data = sensor.get_data()
        detector.add_data(data)
        imu_send_buffer.append(data)  # 버퍼에만 추가
        time.sleep(1.0 / SAMPLING_RATE)
    
    print("🎯 낙상 감지 시작")
    
    # --- IMU 송신 루프 (10Hz) ---
    def imu_send_loop():
        while True:
            if imu_send_buffer:
                latest_data = imu_send_buffer[-1]
                data_sender.add_imu_data(create_imu_package(latest_data, USER_ID))
            time.sleep(1.0 / SEND_RATE)
    
    imu_sender_thread = threading.Thread(target=imu_send_loop, daemon=True)
    imu_sender_thread.start()
    
    # --- 메인 루프 (100Hz) ---
    last_print = time.time()
    
    while True:
        try:
            data = sensor.get_data()
            detector.add_data(data)
            imu_send_buffer.append(data)  # 100Hz로 버퍼에만 추가
            
            # 디버그 출력 (5초마다)
            current_time = time.time()
            if current_time - last_print >= 5.0:
                print(f"가속도: X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                print(f"자이로: X={data[3]:.2f}, Y={data[4]:.2f}, Z={data[5]:.2f}")
                print(f"연결상태: {'✅' if data_sender.connected else '❌'}")
                print(f"샘플링: {SAMPLING_RATE}Hz, 버퍼크기: {len(detector.buffer)}/{SEQ_LENGTH}")
                last_print = current_time
            
            # 낙상 예측
            if detector.should_predict():
                result = detector.predict()
                if result and result['prediction'] == 1:
                    print(f"\n🚨 낙상 감지! 신뢰도: {result['probability']:.2%}")
                    fall_package = create_fall_package(USER_ID, result['probability'], data)
                    data_sender.add_fall_data(fall_package)
                    print("🚨 NAKSANG!")
                    time.sleep(2)
            
            time.sleep(1.0 / SAMPLING_RATE)
        except Exception as e:
            print(f"메인 루프 오류: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()