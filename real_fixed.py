#!/usr/bin/env python3
"""
Real-time Gait Detection System - FIXED VERSION
수정사항:
1. Y축 음수 부호 제거
2. 스케일링 비활성화 (문제가 있을 경우)
3. 모델 출력 디버그 강화
4. 임계값 조정 옵션 추가
"""

import numpy as np
import pickle
import json
import time
import threading
from collections import deque
from datetime import datetime
import tensorflow as tf
import smbus
from bitstring import Bits

class RealTimeGaitDetector:
    def __init__(self, model_path="models/gait_detection/model.tflite", 
                 scaler_path="scalers/gait_detection/standard_scaler.pkl",
                 label_encoder_path="scalers/gait_detection/label_encoder.pkl",
                 thresholds_path="scalers/gait_detection/thresholds.json",
                 use_scaler=False):  # 스케일러 사용 여부 제어
        """
        Initialize real-time gait detection system
        """
        self.window_size = 60
        self.stride = 1
        self.sampling_rate = 30
        self.n_features = 6
        self.use_scaler = use_scaler  # 스케일러 사용 여부
        
        # IMU register addresses
        self.register_gyro_xout_h = 0x43
        self.register_gyro_yout_h = 0x45
        self.register_gyro_zout_h = 0x47
        self.sensitive_gyro = 131.0
        
        self.register_accel_xout_h = 0x3B
        self.register_accel_yout_h = 0x3D
        self.register_accel_zout_h = 0x3F
        self.sensitive_accel = 16384.0
        
        self.DEV_ADDR = 0x68
        
        # Data buffer
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Initialize IMU sensor
        self.init_imu_sensor()
        
        # Load model and preprocessing objects
        self.load_model_and_preprocessors(model_path, scaler_path, 
                                        label_encoder_path, thresholds_path)
        
        # Variables for real-time data collection
        self.is_collecting = False
        self.collection_thread = None
        
        print("✅ Real-time gait detection system initialized successfully")
        print(f"📊 Configuration: window_size={self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate}Hz")
        print(f"🔧 Scaler enabled: {self.use_scaler}")
    
    def init_imu_sensor(self):
        """Initialize IMU sensor (MPU6050) with low-level I2C"""
        try:
            self.bus = smbus.SMBus(1)
            self.bus.write_byte_data(self.DEV_ADDR, 0x6B, 0)
            print("✅ MPU6050 sensor initialized successfully with I2C bus")
            self.sensor_available = True
        except Exception as e:
            print(f"❌ IMU sensor initialization failed: {e}")
            print("💡 Switching to simulation mode.")
            self.sensor_available = False
    
    def read_data(self, register):
        """Read data from IMU register"""
        high = self.bus.read_byte_data(self.DEV_ADDR, register)
        low = self.bus.read_byte_data(self.DEV_ADDR, register+1)
        val = (high << 8) + low
        return val
    
    def twocomplements(self, val):
        """Convert 2's complement"""
        s = Bits(uint=val, length=16)
        return s.int
    
    def gyro_dps(self, val):
        """Convert gyroscope value to degrees/second"""
        return self.twocomplements(val) / self.sensitive_gyro
    
    def accel_ms2(self, val):
        """Convert acceleration value to m/s²"""
        return (self.twocomplements(val) / self.sensitive_accel) * 9.80665
    
    def load_model_and_preprocessors(self, model_path, scaler_path, 
                                   label_encoder_path, thresholds_path):
        """Load model and preprocessing objects"""
        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ TFLite model loaded successfully: {model_path}")
            
            # Load StandardScaler (선택적)
            if self.use_scaler:
                try:
                    with open(scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)
                    print(f"✅ StandardScaler loaded successfully: {scaler_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load StandardScaler: {e}")
                    print("💡 Disabling scaler")
                    self.scaler = None
                    self.use_scaler = False
            else:
                print("🔧 Scaler disabled by configuration")
                self.scaler = None
            
            # Load label encoder
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded successfully: {label_encoder_path}")
            
            # Load thresholds with multiple fallback options
            try:
                with open(thresholds_path, "r") as f:
                    thresholds = json.load(f)
                threshold_values = list(thresholds.values())
                self.optimal_threshold = np.mean(threshold_values)
                print(f"✅ Optimal thresholds loaded: {self.optimal_threshold:.3f}")
            except Exception as e:
                print(f"⚠️ Failed to load thresholds: {e}")
                # 더 높은 임계값들로 테스트
                self.optimal_threshold = 0.5
                print(f"💡 Using default threshold: {self.optimal_threshold}")
            
        except Exception as e:
            print(f"❌ Failed to load model/preprocessing objects: {e}")
            raise
    
    def read_imu_data(self):
        """Read data from IMU sensor - Y축 음수 부호 제거"""
        if not self.sensor_available:
            # 더 현실적인 시뮬레이션 데이터
            accel = [np.random.normal(0, 2) for _ in range(3)]  # 가속도: 평균0, 표준편차2
            gyro = [np.random.normal(0, 50) for _ in range(3)]   # 자이로: 평균0, 표준편차50
        else:
            try:
                # Y축 음수 부호 제거!
                accel_x = self.accel_ms2(self.read_data(self.register_accel_xout_h))
                accel_y = self.accel_ms2(self.read_data(self.register_accel_yout_h))  # 음수 제거
                accel_z = self.accel_ms2(self.read_data(self.register_accel_zout_h))
                
                gyro_x = self.gyro_dps(self.read_data(self.register_gyro_xout_h))
                gyro_y = self.gyro_dps(self.read_data(self.register_gyro_yout_h))
                gyro_z = self.gyro_dps(self.read_data(self.register_gyro_zout_h))
                
                accel = [accel_x, accel_y, accel_z]
                gyro = [gyro_x, gyro_y, gyro_z]
                
            except Exception as e:
                print(f"⚠️ Sensor read error: {e}, using simulation data")
                accel = [np.random.normal(0, 2) for _ in range(3)]
                gyro = [np.random.normal(0, 50) for _ in range(3)]
        
        return accel + gyro
    
    def preprocess_data(self, window_data):
        """Data preprocessing - 스케일링 선택적 적용"""
        X_new = np.array(window_data).reshape(1, self.window_size, self.n_features)
        
        if self.use_scaler and hasattr(self, 'scaler') and self.scaler is not None:
            try:
                n_samples, n_timesteps, n_features = X_new.shape
                X_2d = X_new.reshape(-1, n_features)
                X_scaled = self.scaler.transform(X_2d)
                X_scaled = X_scaled.reshape(X_new.shape)
                
                print(f"🔧 Applied scaling")
                return X_scaled.astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ Scaling failed: {e}, using raw data")
                return X_new.astype(np.float32)
        else:
            print(f"🔧 Using raw data (no scaling)")
            return X_new.astype(np.float32)
    
    def predict_tflite(self, X_preprocessed):
        """Predict using TFLite model with debug info"""
        self.interpreter.set_tensor(self.input_details[0]['index'], X_preprocessed)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 디버그 정보 추가
        if hasattr(self, 'prediction_count'):
            self.prediction_count += 1
        else:
            self.prediction_count = 1
            
        # 처음 몇 번의 예측에서 상세 디버그 정보 출력
        if self.prediction_count <= 3:
            print(f"🔍 Debug prediction #{self.prediction_count}:")
            print(f"    Input shape: {X_preprocessed.shape}")
            print(f"    Input mean: {np.mean(X_preprocessed):.6f}")
            print(f"    Input std: {np.std(X_preprocessed):.6f}")
            print(f"    Input min/max: {np.min(X_preprocessed):.6f}/{np.max(X_preprocessed):.6f}")
            print(f"    Raw output: {output_data}")
        
        return output_data
    
    def collect_data_continuously(self):
        """Continuously collect IMU data (30Hz)"""
        interval = 1.0 / self.sampling_rate
        
        while self.is_collecting:
            start_time = time.time()
            
            sensor_data = self.read_imu_data()
            self.data_buffer.append(sensor_data)
            
            if len(self.data_buffer) == self.window_size:
                self.perform_prediction()
            
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def perform_prediction(self):
        """Perform prediction with enhanced debugging"""
        try:
            window_data = list(self.data_buffer)
            current_sample = window_data[-1]
            accel_x, accel_y, accel_z = current_sample[0], current_sample[1], current_sample[2]
            gyro_x, gyro_y, gyro_z = current_sample[3], current_sample[4], current_sample[5]
            
            # Preprocessing
            X_preprocessed = self.preprocess_data(window_data)
            
            # Prediction
            y_prob = self.predict_tflite(X_preprocessed)
            
            # 다양한 임계값으로 테스트
            thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
            predictions = {}
            for thresh in thresholds_to_test:
                pred = (y_prob > thresh).astype(int)
                label = self.label_encoder.inverse_transform(pred.flatten())[0]
                predictions[thresh] = label
            
            confidence = y_prob[0][0]
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # 기본 임계값으로 예측
            y_pred = (y_prob > self.optimal_threshold).astype(int)
            prediction_label = self.label_encoder.inverse_transform(y_pred.flatten())[0]
            status_icon = "🚶" if prediction_label == "gait" else "🧍"
            
            print(f"[{timestamp}] {status_icon} Main prediction: {prediction_label} "
                  f"(confidence: {confidence:.6f}, threshold: {self.optimal_threshold:.3f})")
            print(f"    📊 Accel: X={accel_x:+7.3f} Y={accel_y:+7.3f} Z={accel_z:+7.3f} m/s²")
            print(f"    🔄 Gyro:  X={gyro_x:+7.3f} Y={gyro_y:+7.3f} Z={gyro_z:+7.3f} °/s")
            
            # 다양한 임계값 결과 표시
            thresh_results = " | ".join([f"{thresh}:{predictions[thresh][:4]}" for thresh in thresholds_to_test])
            print(f"    🎯 Threshold tests: {thresh_results}")
            
            # 문제 상황 감지
            if confidence == 1.0 or confidence == 0.0:
                print(f"    ⚠️  WARNING: Confidence is exactly {confidence} - model might be saturated!")
            
            print()
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_detection(self, show_sensor_details=True):
        """Start real-time gait detection"""
        if self.is_collecting:
            print("⚠️ Detection is already running.")
            return
        
        print("🎯 Starting real-time gait detection...")
        print("📋 Legend: 🚶 = Walking, 🧍 = Standing")
        print("⏹️ Press Ctrl+C to stop.")
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self.collect_data_continuously)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        try:
            while self.is_collecting:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_detection()
    
    def stop_detection(self):
        """Stop real-time gait detection"""
        print("\n🛑 Stopping real-time gait detection...")
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        print("✅ Gait detection stopped successfully.")

def main():
    """Main function with configuration options"""
    print("🤖 Raspberry Pi 4 Real-time Gait Detection System - FIXED VERSION")
    print("=" * 60)
    
    try:
        # 스케일러 없이 먼저 테스트
        print("🔧 Testing without scaler first...")
        detector = RealTimeGaitDetector(use_scaler=False)
        detector.start_detection()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Exiting program.")

if __name__ == "__main__":
    main()