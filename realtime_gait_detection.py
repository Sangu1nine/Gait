#!/usr/bin/env python3
"""
실시간 보행 감지 시스템 - 라즈베리파이4
30Hz IMU 센서, 60 window, 1 stride

MODIFIED [2024-12-19]: 초기 구현 - 실시간 IMU 데이터 수집 및 보행 감지
"""

import numpy as np
import pickle
import json
import time
import threading
from collections import deque
from datetime import datetime
import tensorflow as tf
import board
import busio
import adafruit_mpu6050

class RealTimeGaitDetector:
    def __init__(self, model_path="models/gait_detection/model.tflite", 
                 scaler_path="scalers/gait_detection/standard_scaler.pkl",
                 label_encoder_path="scalers/gait_detection/label_encoder.pkl",
                 thresholds_path="scalers/gait_detection/thresholds.json"):
        """
        실시간 보행 감지 시스템 초기화
        
        Args:
            model_path: TFLite 모델 경로
            scaler_path: StandardScaler 경로
            label_encoder_path: 라벨 인코더 경로
            thresholds_path: 최적 임계값 경로
        """
        self.window_size = 60  # 60 샘플 윈도우
        self.stride = 1        # 1 샘플 스트라이드
        self.sampling_rate = 30  # 30Hz
        self.n_features = 6    # 가속도 3축 + 자이로 3축
        
        # 데이터 버퍼 (60개 샘플 저장)
        self.data_buffer = deque(maxlen=self.window_size)
        
        # IMU 센서 초기화
        self.init_imu_sensor()
        
        # 모델 및 전처리 객체 로드
        self.load_model_and_preprocessors(model_path, scaler_path, 
                                        label_encoder_path, thresholds_path)
        
        # 실시간 데이터 수집을 위한 변수
        self.is_collecting = False
        self.collection_thread = None
        
        print("✅ 실시간 보행 감지 시스템 초기화 완료")
        print(f"📊 설정: 윈도우 크기={self.window_size}, 스트라이드={self.stride}, 샘플링 주파수={self.sampling_rate}Hz")
    
    def init_imu_sensor(self):
        """IMU 센서 (MPU6050) 초기화"""
        try:
            # I2C 인터페이스 초기화
            i2c = busio.I2C(board.SCL, board.SDA)
            self.mpu = adafruit_mpu6050.MPU6050(i2c)
            
            # 센서 설정
            self.mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_4_G
            self.mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_500_DPS
            
            print("✅ MPU6050 센서 초기화 완료")
            
        except Exception as e:
            print(f"❌ IMU 센서 초기화 실패: {e}")
            print("💡 시뮬레이션 모드로 전환됩니다.")
            self.mpu = None
    
    def load_model_and_preprocessors(self, model_path, scaler_path, 
                                   label_encoder_path, thresholds_path):
        """모델 및 전처리 객체 로드"""
        try:
            # TFLite 모델 로드
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # 입력/출력 정보 가져오기
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ TFLite 모델 로드 완료: {model_path}")
            
            # StandardScaler 로드
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✅ StandardScaler 로드 완료: {scaler_path}")
            
            # 라벨 인코더 로드
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ 라벨 인코더 로드 완료: {label_encoder_path}")
            
            # 최적 임계값 로드
            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)
            
            # 임계값 평균 계산
            threshold_values = list(thresholds.values())
            self.optimal_threshold = np.mean(threshold_values)
            print(f"✅ 최적 임계값 로드 완료: {self.optimal_threshold:.3f}")
            print(f"📋 Fold별 임계값: {thresholds}")
            
        except Exception as e:
            print(f"❌ 모델/전처리 객체 로드 실패: {e}")
            raise
    
    def read_imu_data(self):
        """IMU 센서에서 데이터 읽기"""
        if self.mpu is None:
            # 시뮬레이션 데이터 (테스트용)
            accel = [np.random.normal(0, 1) for _ in range(3)]
            gyro = [np.random.normal(0, 1) for _ in range(3)]
        else:
            # 실제 센서 데이터
            accel = list(self.mpu.acceleration)  # m/s²
            gyro = list(self.mpu.gyro)          # rad/s
        
        return accel + gyro  # 6개 특성 반환
    
    def preprocess_data(self, window_data):
        """데이터 전처리 (README.md 참고)"""
        # numpy 배열로 변환 (1, 60, 6)
        X = np.array(window_data).reshape(1, self.window_size, self.n_features)
        
        # 3D -> 2D 변환
        X_2d = X.reshape(-1, self.n_features)
        
        # StandardScaler 적용 (fit 없이 transform만!)
        X_scaled = self.scaler.transform(X_2d)
        
        # 2D -> 3D 변환
        X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled.astype(np.float32)
    
    def predict_tflite(self, X_preprocessed):
        """TFLite 모델로 예측"""
        # 입력 데이터 설정
        self.interpreter.set_tensor(self.input_details[0]['index'], X_preprocessed)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 결과 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def collect_data_continuously(self):
        """지속적으로 IMU 데이터 수집 (30Hz)"""
        interval = 1.0 / self.sampling_rate  # 30Hz = 33.33ms 간격
        
        while self.is_collecting:
            start_time = time.time()
            
            # IMU 데이터 읽기
            sensor_data = self.read_imu_data()
            
            # 버퍼에 추가
            self.data_buffer.append(sensor_data)
            
            # 윈도우가 채워졌으면 예측 수행
            if len(self.data_buffer) == self.window_size:
                self.perform_prediction()
            
            # 정확한 샘플링 주파수 유지
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def perform_prediction(self):
        """현재 윈도우에 대해 예측 수행"""
        try:
            # 윈도우 데이터 복사
            window_data = list(self.data_buffer)
            
            # 전처리
            X_preprocessed = self.preprocess_data(window_data)
            
            # 예측
            y_prob = self.predict_tflite(X_preprocessed)
            
            # 임계값 적용
            y_pred = (y_prob > self.optimal_threshold).astype(int)
            
            # 라벨 디코딩
            prediction_label = self.label_encoder.inverse_transform(y_pred.flatten())[0]
            confidence = y_prob[0][0]
            
            # 결과 출력
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            status_icon = "🚶" if prediction_label == "gait" else "🧍"
            
            print(f"[{timestamp}] {status_icon} 예측: {prediction_label} "
                  f"(신뢰도: {confidence:.3f}, 임계값: {self.optimal_threshold:.3f})")
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
    
    def start_detection(self):
        """실시간 보행 감지 시작"""
        if self.is_collecting:
            print("⚠️ 이미 감지가 실행 중입니다.")
            return
        
        print("🎯 실시간 보행 감지를 시작합니다...")
        print("📋 범례: 🚶 = 보행 중, 🧍 = 정지 상태")
        print("⏹️ 중지하려면 Ctrl+C를 누르세요.")
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self.collect_data_continuously)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        try:
            # 메인 스레드가 종료되지 않도록 대기
            while self.is_collecting:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_detection()
    
    def stop_detection(self):
        """실시간 보행 감지 중지"""
        print("\n🛑 실시간 보행 감지를 중지합니다...")
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        print("✅ 보행 감지가 중지되었습니다.")

def main():
    """메인 함수"""
    print("🤖 라즈베리파이4 실시간 보행 감지 시스템")
    print("=" * 50)
    
    try:
        # 보행 감지 시스템 초기화
        detector = RealTimeGaitDetector()
        
        # 실시간 감지 시작
        detector.start_detection()
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("💡 모델 파일과 전처리 객체 파일들이 올바른 경로에 있는지 확인하세요.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    finally:
        print("👋 프로그램을 종료합니다.")

if __name__ == "__main__":
    main() 