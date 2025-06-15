#!/usr/bin/env python3
"""
Real-time Gait Detection System - Raspberry Pi 4
30Hz IMU sensor, 60 window, 1 stride

MODIFIED [2024-12-19]: Initial implementation - Real-time IMU data collection and gait detection
MODIFIED [2024-12-19]: Updated IMU sensor data reading with low-level I2C communication
MODIFIED [2024-12-19]: Added IMU sensor reading debugging and improved data reading methods
MODIFIED [2024-12-19]: Applied successful sensor reading method from gait_data_30hz.py (smbus2 + burst read)
"""

import numpy as np
import pickle
import json
import time
import threading
from collections import deque
from datetime import datetime
import tensorflow as tf
from smbus2 import SMBus  # Changed from smbus to smbus2
from bitstring import Bits

class RealTimeGaitDetector:
    def __init__(self, model_path="models/gait_detection/model.tflite", 
                 scaler_path="scalers/gait_detection/standard_scaler.pkl",
                 label_encoder_path="scalers/gait_detection/label_encoder.pkl",
                 thresholds_path="scalers/gait_detection/thresholds.json"):
        """
        Initialize real-time gait detection system
        
        Args:
            model_path: TFLite model path
            scaler_path: StandardScaler path
            label_encoder_path: Label encoder path
            thresholds_path: Optimal thresholds path
        """
        self.window_size = 60  # 60 sample window
        self.stride = 1        # 1 sample stride
        self.sampling_rate = 30  # 30Hz
        self.n_features = 6    # 3-axis accelerometer + 3-axis gyroscope
        
        # IMU register addresses - 확인된 주소들
        self.register_gyro_xout_h = 0x43
        self.register_gyro_yout_h = 0x45
        self.register_gyro_zout_h = 0x47
        self.sensitive_gyro = 131.0
        
        self.register_accel_xout_h = 0x3B
        self.register_accel_yout_h = 0x3D  # ACCEL_YOUT_H 확인
        self.register_accel_yout_l = 0x3E  # ACCEL_YOUT_L 추가
        self.register_accel_zout_h = 0x3F
        self.sensitive_accel = 16384.0
        
        self.DEV_ADDR = 0x68  # MPU6050 I2C address
        
        # Data buffer (stores 60 samples)
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
    
    def init_imu_sensor(self):
        """Initialize IMU sensor (MPU6050) with low-level I2C"""
        try:
            # Initialize I2C bus
            self.bus = SMBus(1)  # Use I2C bus 1
            
            # Wake up MPU6050 (reset sleep mode)
            self.bus.write_byte_data(self.DEV_ADDR, 0x6B, 0)
            
            print("✅ MPU6050 sensor initialized successfully with I2C bus")
            self.sensor_available = True
            
            # 센서 진단 실행
            self.diagnose_accel_reading()
            
        except Exception as e:
            print(f"❌ IMU sensor initialization failed: {e}")
            print("💡 Switching to simulation mode.")
            self.sensor_available = False
    
    def read_data(self, register):
        """Read data from IMU register (original method)"""
        high = self.bus.read_byte_data(self.DEV_ADDR, register)
        low = self.bus.read_byte_data(self.DEV_ADDR, register+1)
        val = (high << 8) + low
        return val
    
    def read_data_debug(self, register):
        """디버깅을 위한 상세 출력을 포함한 데이터 읽기"""
        high = self.bus.read_byte_data(self.DEV_ADDR, register)
        low = self.bus.read_byte_data(self.DEV_ADDR, register+1)
        
        # 빅 엔디안 방식
        val_big = (high << 8) | low
        
        # 리틀 엔디안 방식 (테스트용)
        val_little = (low << 8) | high
        
        print(f"Register 0x{register:02X} - High byte: 0x{high:02X}, Low byte: 0x{low:02X}")
        print(f"빅 엔디안: {val_big} (0x{val_big:04X})")
        print(f"리틀 엔디안: {val_little} (0x{val_little:04X})")
        
        return val_big
    
    def read_data_word(self, register):
        """SMBus의 word 읽기 함수 사용"""
        # SMBus는 리틀 엔디안으로 읽으므로 바이트 스왑 필요
        word = self.bus.read_word_data(self.DEV_ADDR, register)
        # 바이트 스왑 (리틀 → 빅 엔디안)
        swapped = ((word & 0xFF) << 8) | ((word >> 8) & 0xFF)
        return swapped
    
    def diagnose_accel_reading(self):
        """가속도계 읽기 진단"""
        if not self.sensor_available:
            return
            
        print("\n=== 가속도계 Y축 읽기 진단 ===")
        
        try:
            # Test burst reading first
            print("🔍 버스트 읽기 테스트:")
            try:
                sensor_data = self.read_data_burst()
                print(f"✅ 버스트 읽기 성공!")
                print(f"   Accel: X={sensor_data[0]:.3f}, Y={sensor_data[1]:.3f}, Z={sensor_data[2]:.3f} m/s²")
                print(f"   Gyro: X={sensor_data[3]:.3f}, Y={sensor_data[4]:.3f}, Z={sensor_data[5]:.3f} °/s")
            except Exception as e:
                print(f"❌ 버스트 읽기 실패: {e}")
            
            # 1. 현재 방식
            high = self.bus.read_byte_data(self.DEV_ADDR, 0x3D)
            low = self.bus.read_byte_data(self.DEV_ADDR, 0x3E)
            val_current = (high << 8) + low
            
            # 2. word 읽기 방식
            word = self.bus.read_word_data(self.DEV_ADDR, 0x3D)
            val_word_swap = ((word & 0xFF) << 8) | ((word >> 8) & 0xFF)
            
            # 3. 2의 보수 변환
            signed_current = self.twocomplements(val_current)
            signed_word = self.twocomplements(val_word_swap)
            
            # 4. 최종 값 계산
            accel_current = (signed_current / 16384.0) * 9.80665
            accel_word = (signed_word / 16384.0) * 9.80665
            
            print(f"현재 방식: {accel_current:.3f} m/s²")
            print(f"Word 방식: {accel_word:.3f} m/s²")
            
            # 5. 센서 설정 확인
            config = self.bus.read_byte_data(self.DEV_ADDR, 0x1C)
            range_setting = (config >> 3) & 0x03
            range_labels = ['±2g', '±4g', '±8g', '±16g']
            print(f"가속도계 범위 설정: {range_labels[range_setting]}")
            
            # 6. 모든 축 테스트
            print("\n=== 전체 가속도계 축 테스트 ===")
            for axis, reg in [('X', 0x3B), ('Y', 0x3D), ('Z', 0x3F)]:
                high = self.bus.read_byte_data(self.DEV_ADDR, reg)
                low = self.bus.read_byte_data(self.DEV_ADDR, reg+1)
                val = (high << 8) + low
                signed_val = self.twocomplements(val)
                accel = (signed_val / 16384.0) * 9.80665
                print(f"{axis}축: Raw={val:5d} (0x{val:04X}), Signed={signed_val:6d}, Accel={accel:+7.3f} m/s²")
                
        except Exception as e:
            print(f"❌ 진단 중 오류 발생: {e}")
            
        print("=== 진단 완료 ===\n")
    
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
            
            # Get input/output information
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ TFLite model loaded successfully: {model_path}")
            
            # Try to load StandardScaler
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print(f"✅ StandardScaler loaded successfully: {scaler_path}")
                
                # Check scaler properties
                if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                    print(f"🔍 Scaler mean: {self.scaler.mean_}")
                    print(f"🔍 Scaler scale: {self.scaler.scale_}")
                else:
                    print("⚠️ Scaler doesn't have expected attributes")
                    
            except Exception as e:
                print(f"⚠️ Failed to load StandardScaler: {e}")
                print("💡 Will proceed without scaling")
                self.scaler = None
            
            # Load label encoder
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded successfully: {label_encoder_path}")
            
            # Try to load optimal thresholds, but use 0.5 as fallback (per README.md)
            try:
                with open(thresholds_path, "r") as f:
                    thresholds = json.load(f)
                
                # Calculate threshold average
                threshold_values = list(thresholds.values())
                self.optimal_threshold = np.mean(threshold_values)
                print(f"✅ Optimal thresholds loaded successfully: {self.optimal_threshold:.3f}")
                print(f"📋 Fold-wise thresholds: {thresholds}")
                
            except Exception as e:
                print(f"⚠️ Failed to load thresholds: {e}")
                print("💡 Using default threshold from README.md: 0.5")
                self.optimal_threshold = 0.24
            
        except Exception as e:
            print(f"❌ Failed to load model/preprocessing objects: {e}")
            raise
    
    def read_imu_data(self, use_word_method=False, debug_mode=False):
        """
        Read data from IMU sensor with proper preprocessing
        Using successful burst read method from gait_data_30hz.py as primary method
        
        Args:
            use_word_method: SMBus word 읽기 방법 사용 여부 (deprecated - burst read is now primary)
            debug_mode: 디버깅 모드 (상세 출력)
        """
        if not self.sensor_available:
            # Simulation data (for testing)
            accel = [np.random.normal(0, 1) for _ in range(3)]
            gyro = [np.random.normal(0, 1) for _ in range(3)]
        else:
            try:
                # Primary method: Use burst read (same as gait_data_30hz.py)
                if debug_mode:
                    print("🔍 Using burst read method (gait_data_30hz.py style)")
                
                sensor_data = self.read_data_burst()
                accel = sensor_data[:3]
                gyro = sensor_data[3:]
                
                if debug_mode:
                    print(f"Burst read success - Accel: X={accel[0]:.3f}, Y={accel[1]:.3f}, Z={accel[2]:.3f} m/s²")
                    print(f"Burst read success - Gyro: X={gyro[0]:.3f}, Y={gyro[1]:.3f}, Z={gyro[2]:.3f} °/s")
                
            except Exception as burst_error:
                if debug_mode:
                    print(f"⚠️ Burst read failed: {burst_error}, falling back to individual reads")
                
                try:
                    if use_word_method:
                        # SMBus word 읽기 방법 사용
                        if debug_mode:
                            print("🔍 Using SMBus word reading method")
                        
                        accel_x_raw = self.read_data_word(self.register_accel_xout_h)
                        accel_y_raw = self.read_data_word(self.register_accel_yout_h)
                        accel_z_raw = self.read_data_word(self.register_accel_zout_h)
                        
                        gyro_x_raw = self.read_data_word(self.register_gyro_xout_h)
                        gyro_y_raw = self.read_data_word(self.register_gyro_yout_h)
                        gyro_z_raw = self.read_data_word(self.register_gyro_zout_h)
                        
                    else:
                        # 기존 방법 사용
                        if debug_mode:
                            print("🔍 Using original byte reading method")
                        
                        accel_x_raw = self.read_data(self.register_accel_xout_h)
                        accel_y_raw = self.read_data(self.register_accel_yout_h)
                        accel_z_raw = self.read_data(self.register_accel_zout_h)
                        
                        gyro_x_raw = self.read_data(self.register_gyro_xout_h)
                        gyro_y_raw = self.read_data(self.register_gyro_yout_h)
                        gyro_z_raw = self.read_data(self.register_gyro_zout_h)
                    
                    # 단위 변환
                    accel_x = self.accel_ms2(accel_x_raw)
                    accel_y = -self.accel_ms2(accel_y_raw)  # Note: negative sign
                    accel_z = self.accel_ms2(accel_z_raw)
                    
                    gyro_x = self.gyro_dps(gyro_x_raw)
                    gyro_y = self.gyro_dps(gyro_y_raw)
                    gyro_z = self.gyro_dps(gyro_z_raw)
                    
                    accel = [accel_x, accel_y, accel_z]
                    gyro = [gyro_x, gyro_y, gyro_z]
                    
                    if debug_mode:
                        print(f"Raw values - Accel: X={accel_x_raw}, Y={accel_y_raw}, Z={accel_z_raw}")
                        print(f"Raw values - Gyro: X={gyro_x_raw}, Y={gyro_y_raw}, Z={gyro_z_raw}")
                        print(f"Converted - Accel: X={accel_x:.3f}, Y={accel_y:.3f}, Z={accel_z:.3f} m/s²")
                        print(f"Converted - Gyro: X={gyro_x:.3f}, Y={gyro_y:.3f}, Z={gyro_z:.3f} °/s")
                        
                except Exception as fallback_error:
                    print(f"⚠️ All sensor read methods failed: {fallback_error}, using simulation data")
                    accel = [np.random.normal(0, 1) for _ in range(3)]
                    gyro = [np.random.normal(0, 1) for _ in range(3)]
        
        return accel + gyro  # Return 6 features: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    def preprocess_data(self, window_data):
        """Data preprocessing following README.md guide exactly"""
        # Convert to numpy array (n_samples, 60, 6) - in our case n_samples=1
        X_new = np.array(window_data).reshape(1, self.window_size, self.n_features)
        
        # Check if we should apply scaling
        if hasattr(self, 'scaler') and self.scaler is not None:
            try:
                # Following README.md exactly:
                n_samples, n_timesteps, n_features = X_new.shape
                
                # 3D -> 2D 변환
                X_2d = X_new.reshape(-1, n_features)
                
                # 스케일링 적용 (fit 없이 transform만!)
                X_scaled = self.scaler.transform(X_2d)
                
                # 2D -> 3D 변환
                X_scaled = X_scaled.reshape(X_new.shape)
                
                return X_scaled.astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ Scaling failed: {e}")
                print("💡 Using raw data without scaling")
                return X_new.astype(np.float32)
        else:
            print("⚠️ No scaler available, using raw data")
            return X_new.astype(np.float32)
    
    def predict_tflite(self, X_preprocessed):
        """Predict using TFLite model"""
        # Set input data
        self.interpreter.set_tensor(self.input_details[0]['index'], X_preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def test_reading_methods(self, samples=10):
        """여러 읽기 방법을 비교 테스트"""
        if not self.sensor_available:
            print("❌ 센서를 사용할 수 없습니다.")
            return
            
        print(f"\n=== 읽기 방법 비교 테스트 ({samples}회) ===")
        
        original_results = []
        word_results = []
        
        for i in range(samples):
            print(f"\n테스트 {i+1}/{samples}")
            
            # 기존 방법
            data_original = self.read_imu_data(use_word_method=False, debug_mode=True)
            original_results.append(data_original)
            
            time.sleep(0.1)  # 잠시 대기
            
            # Word 방법
            data_word = self.read_imu_data(use_word_method=True, debug_mode=True)
            word_results.append(data_word)
            
            time.sleep(0.1)  # 잠시 대기
        
        # 결과 분석
        orig_arr = np.array(original_results)
        word_arr = np.array(word_results)
        
        print(f"\n=== 결과 분석 ===")
        print(f"기존 방법 - 평균: {np.mean(orig_arr, axis=0)}")
        print(f"기존 방법 - 표준편차: {np.std(orig_arr, axis=0)}")
        print(f"Word 방법 - 평균: {np.mean(word_arr, axis=0)}")
        print(f"Word 방법 - 표준편차: {np.std(word_arr, axis=0)}")
        
        # 차이 계산
        diff = np.abs(orig_arr - word_arr)
        print(f"절대 차이 - 평균: {np.mean(diff, axis=0)}")
        print(f"절대 차이 - 최대: {np.max(diff, axis=0)}")
    
    def start_detection(self, show_sensor_details=True, use_word_method=False, debug_mode=False):
        """
        Start real-time gait detection
        
        Args:
            show_sensor_details: 센서 상세 데이터 표시 여부
            use_word_method: SMBus word 읽기 방법 사용 여부
            debug_mode: 디버깅 모드
        """
        if self.is_collecting:
            print("⚠️ Detection is already running.")
            return
        
        # 읽기 방법 설정 저장
        self.use_word_method = use_word_method
        self.debug_mode = debug_mode
        
        print("🎯 Starting real-time gait detection...")
        print("📋 Legend: 🚶 = Walking, 🧍 = Standing")
        
        # 실제 사용될 읽기 방법 표시
        if not self.sensor_available:
            print("🔧 Reading method: Simulation Mode (No sensor)")
        else:
            print(f"🔧 Reading method: Burst Read (gait_data_30hz.py style) with fallback to {'SMBus Word' if use_word_method else 'Original Byte'}")
        
        if show_sensor_details:
            print("📊 Sensor data will be displayed with each prediction")
        if debug_mode:
            print("🐛 Debug mode enabled")
        print("⏹️ Press Ctrl+C to stop.")
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self.collect_data_continuously)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_collecting:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_detection()
    
    def collect_data_continuously(self):
        """Continuously collect IMU data (30Hz)"""
        interval = 1.0 / self.sampling_rate  # 30Hz = 33.33ms interval
        
        while self.is_collecting:
            start_time = time.time()
            
            # Read IMU data with selected method
            sensor_data = self.read_imu_data(
                use_word_method=getattr(self, 'use_word_method', False),
                debug_mode=getattr(self, 'debug_mode', False)
            )
            
            # Add to buffer
            self.data_buffer.append(sensor_data)
            
            # Perform prediction when window is filled
            if len(self.data_buffer) == self.window_size:
                self.perform_prediction()
            
            # Maintain accurate sampling frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def perform_prediction(self):
        """Perform prediction on current window"""
        try:
            # Copy window data
            window_data = list(self.data_buffer)
            
            # Get current sensor values (latest sample)
            current_sample = window_data[-1]
            accel_x, accel_y, accel_z = current_sample[0], current_sample[1], current_sample[2]
            gyro_x, gyro_y, gyro_z = current_sample[3], current_sample[4], current_sample[5]
            
            # Show some preprocessing debug info
            window_array = np.array(window_data)
            raw_mean = np.mean(window_array, axis=0)
            raw_std = np.std(window_array, axis=0)
            
            # Preprocessing
            X_preprocessed = self.preprocess_data(window_data)
            
            # Show preprocessed data statistics
            X_flat = X_preprocessed.reshape(-1, self.n_features)
            scaled_mean = np.mean(X_flat, axis=0)
            scaled_std = np.std(X_flat, axis=0)
            
            # Prediction
            y_prob = self.predict_tflite(X_preprocessed)
            
            # Apply threshold
            y_pred = (y_prob > self.optimal_threshold).astype(int)
            
            # Decode labels
            prediction_label = self.label_encoder.inverse_transform(y_pred.flatten())[0]
            confidence = y_prob[0][0]
            
            # Display results with sensor values
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            status_icon = "🚶" if prediction_label == "gait" else "🧍"
            
            print(f"[{timestamp}] {status_icon} Prediction: {prediction_label} "
                  f"(confidence: {confidence:.3f}, threshold: {self.optimal_threshold:.3f})")
            print(f"    📊 Accel: X={accel_x:+7.3f} Y={accel_y:+7.3f} Z={accel_z:+7.3f} m/s²")
            print(f"    🔄 Gyro:  X={gyro_x:+7.3f} Y={gyro_y:+7.3f} Z={gyro_z:+7.3f} °/s")
            print(f"    🎯 Raw probability: {y_prob[0][0]:.6f}")
            
            # Debug preprocessing data
            print(f"    🔍 Raw data mean: Acc({raw_mean[0]:+5.2f},{raw_mean[1]:+5.2f},{raw_mean[2]:+5.2f}) "
                  f"Gyro({raw_mean[3]:+5.2f},{raw_mean[4]:+5.2f},{raw_mean[5]:+5.2f})")
            print(f"    🔍 Scaled data mean: Acc({scaled_mean[0]:+5.2f},{scaled_mean[1]:+5.2f},{scaled_mean[2]:+5.2f}) "
                  f"Gyro({scaled_mean[3]:+5.2f},{scaled_mean[4]:+5.2f},{scaled_mean[5]:+5.2f})")
            print(f"    🔍 Scaled data std: Acc({scaled_std[0]:+5.2f},{scaled_std[1]:+5.2f},{scaled_std[2]:+5.2f}) "
                  f"Gyro({scaled_std[3]:+5.2f},{scaled_std[4]:+5.2f},{scaled_std[5]:+5.2f})")
            
            # Check if scaled data looks normal (should be roughly mean=0, std=1 for each feature)
            if np.any(np.abs(scaled_mean) > 5) or np.any(scaled_std > 10):
                print(f"    ⚠️  Scaling might be problematic - values seem too extreme")
            
            print()  # Add blank line for readability
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_detection(self):
        """Stop real-time gait detection"""
        print("\n🛑 Stopping real-time gait detection...")
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        print("✅ Gait detection stopped successfully.")

def main():
    """Main function"""
    print("🤖 Raspberry Pi 4 Real-time Gait Detection System")
    print("=" * 50)
    
    try:
        # Initialize gait detection system
        detector = RealTimeGaitDetector()
        
        # 사용자 선택 메뉴
        print("\n📋 실행 모드 선택:")
        print("1. 기본 실시간 감지 (Burst 읽기 방법 - 권장)")
        print("2. 개선된 실시간 감지 (SMBus Word 방법)")
        print("3. 디버그 모드 실시간 감지")
        print("4. 센서 읽기 방법 비교 테스트")
        print("5. 센서 진단만 실행")
        
        try:
            choice = input("\n선택하세요 (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"  # 기본값
            print("기본 모드로 실행합니다.")
        
        if choice == "1":
            # 기본 실시간 감지 (Burst 읽기 방법)
            detector.start_detection(show_sensor_details=True, use_word_method=False)
            
        elif choice == "2":
            # 개선된 실시간 감지 (SMBus Word 방법)
            detector.start_detection(show_sensor_details=True, use_word_method=True)
            
        elif choice == "3":
            # 디버그 모드 (Burst 읽기 방법 + 디버그)
            detector.start_detection(show_sensor_details=True, use_word_method=False, debug_mode=True)
            
        elif choice == "4":
            # 비교 테스트
            detector.test_reading_methods(samples=5)
            
        elif choice == "5":
            # 센서 진단만
            if detector.sensor_available:
                detector.diagnose_accel_reading()
            else:
                print("❌ 센서를 사용할 수 없습니다.")
        else:
            print("잘못된 선택입니다. 기본 모드로 실행합니다.")
            detector.start_detection(show_sensor_details=True, use_word_method=False)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Please check if model files and preprocessing object files are in the correct paths.")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Exiting program.")

if __name__ == "__main__":
    main() 