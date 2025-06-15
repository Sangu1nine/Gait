import os
import numpy as np
import pickle
import time
import datetime
import threading
from collections import deque
import json
from smbus2 import SMBus
from bitstring import Bits
import math
from scipy import signal
import tensorflow as tf

# ========== 센서 설정 ==========
bus = SMBus(1)
DEV_ADDR = 0x68

# 레지스터 주소
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0

register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0

# ========== 모델 및 전처리 설정 ==========
MODEL_PATH = "models/gait_detection/model.tflite"  # TFLite 모델 경로
SCALER_PATH = "scalers/gait_detection/standard_scaler.pkl"  # 또는 "minmax_scaler.pkl"
ENCODER_PATH = "scalers/gait_detection/label_encoder.pkl"
THRESHOLD = 0.3  # 예측 임계값

# ========== 실시간 예측 설정 ==========
WINDOW_SIZE = 60  # 2초 (30Hz * 2초)
SAMPLING_RATE = 30  # 30Hz
STRIDE = 1  # 1 frame씩 업데이트
FILTER_ORDER = 4
CUTOFF_FREQ = 10  # 10Hz 컷오프

class RealTimeGaitPredictor:
    def __init__(self, model_path, scaler_path, encoder_path, threshold=0.3):
        """실시간 보행 패턴 예측기 초기화"""
        self.threshold = threshold
        self.window_size = WINDOW_SIZE
        self.sampling_rate = SAMPLING_RATE
        
        # 센서 데이터 윈도우 (60 frames)
        self.sensor_window = deque(maxlen=self.window_size)
        
        # 버터워스 필터 설정
        self.setup_filter()
        
        # 모델 및 전처리 객체 로드
        self.load_model_and_preprocessors(model_path, scaler_path, encoder_path)
        
        # 예측 결과 저장
        self.predictions = []
        self.prediction_lock = threading.Lock()
        
        print("🚀 Real-time gait pattern predictor initialized successfully")
        print(f"   - Window size: {self.window_size} frames (2 seconds)")
        print(f"   - Update interval: Every {STRIDE} frame (1/30 second)")
        print(f"   - Prediction threshold: {self.threshold}")
        print(f"   - Filter: Butterworth {FILTER_ORDER}th order, {CUTOFF_FREQ}Hz")
    
    def setup_filter(self):
        """버터워스 필터 설정"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = CUTOFF_FREQ / nyquist
        self.filter_b, self.filter_a = signal.butter(
            FILTER_ORDER, normal_cutoff, btype='low', analog=False)
        
        # 필터 상태 초기화 (각 센서 채널별)
        self.filter_zi = [signal.lfilter_zi(self.filter_b, self.filter_a) for _ in range(6)]
    
    def apply_filter(self, sensor_data):
        """실시간 버터워스 필터 적용"""
        filtered_data = np.zeros_like(sensor_data)
        
        for i in range(6):  # 6개 센서 채널
            filtered_data[i], self.filter_zi[i] = signal.lfilter(
                self.filter_b, self.filter_a, [sensor_data[i]], zi=self.filter_zi[i])
        
        return filtered_data.flatten()
    
    def load_model_and_preprocessors(self, model_path, scaler_path, encoder_path):
        """모델 및 전처리 객체 로드"""
        print("📁 Loading model and preprocessing objects...")
        
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Input/output information
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"   ✅ TFLite model loaded: {model_path}")
            print(f"      Input shape: {self.input_details[0]['shape']}")
            print(f"      Output shape: {self.output_details[0]['shape']}")
        except Exception as e:
            print(f"   ❌ TFLite model loading failed: {e}")
            raise
        
        # Load scaler
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"   ✅ Scaler loaded: {scaler_path}")
        except Exception as e:
            print(f"   ❌ Scaler loading failed: {e}")
            raise
        
        # Load label encoder
        try:
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"   ✅ Label encoder loaded: {encoder_path}")
            print(f"      Classes: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"   ❌ Label encoder loading failed: {e}")
            raise
    
    def preprocess_window(self, window_data):
        """윈도우 데이터 전처리"""
        # numpy 배열로 변환
        window_array = np.array(window_data, dtype=np.float32)
        
        # 형태 확인: (60, 6)
        if window_array.shape != (self.window_size, 6):
            return None
        
        # 3D -> 2D 변환 (스케일링을 위해)
        window_2d = window_array.reshape(-1, 6)  # (60*1, 6) = (60, 6)
        
        # 스케일링 적용
        window_scaled = self.scaler.transform(window_2d)
        
        # 다시 3D로 변환 후 배치 차원 추가
        window_scaled = window_scaled.reshape(1, self.window_size, 6)  # (1, 60, 6)
        
        return window_scaled.astype(np.float32)
    
    def predict(self, window_data):
        """윈도우 데이터로 예측 수행"""
        try:
            # 전처리
            preprocessed_data = self.preprocess_window(window_data)
            if preprocessed_data is None:
                return None, None
            
            # TFLite 추론
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_data)
            self.interpreter.invoke()
            prediction_prob = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 확률값 (sigmoid 출력)
            prob = float(prediction_prob[0][0])
            
            # 임계값 적용
            predicted_label = 1 if prob > self.threshold else 0
            predicted_class = self.label_encoder.inverse_transform([predicted_label])[0]
            
            return predicted_class, prob
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def add_sensor_data(self, sensor_data):
        """새로운 센서 데이터 추가 및 예측"""
        # 필터 적용
        filtered_data = self.apply_filter(sensor_data)
        
        # 윈도우에 데이터 추가
        self.sensor_window.append(filtered_data)
        
        # 윈도우가 가득 찬 경우에만 예측
        if len(self.sensor_window) == self.window_size:
            predicted_class, probability = self.predict(list(self.sensor_window))
            
            if predicted_class is not None:
                timestamp = time.time()
                
                with self.prediction_lock:
                    prediction_result = {
                        'timestamp': timestamp,
                        'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'predicted_class': predicted_class,
                        'probability': probability,
                        'confidence': probability if predicted_class == 'gait' else (1 - probability)
                    }
                    self.predictions.append(prediction_result)
                
                return prediction_result
        
        return None
    
    def get_recent_predictions(self, n=10):
        """최근 n개의 예측 결과 반환"""
        with self.prediction_lock:
            return self.predictions[-n:] if len(self.predictions) > n else self.predictions.copy()
    
    def save_predictions(self, filename=None):
        """예측 결과를 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gait_predictions_{timestamp}.json"
        
        with self.prediction_lock:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.predictions, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Prediction results saved: {filename} ({len(self.predictions)} predictions)")

# ========== 센서 관련 함수들 ==========
def read_data(register):
    """센서 레지스터에서 데이터 읽기"""
    high = bus.read_byte_data(DEV_ADDR, register)
    low = bus.read_byte_data(DEV_ADDR, register + 1)
    val = (high << 8) + low
    return val

def twocomplements(val):
    """2의 보수 변환"""
    s = Bits(uint=val, length=16)
    return s.int

def gyro_dps(val):
    """자이로스코프 값을 degrees/second로 변환"""
    return twocomplements(val) / sensitive_gyro

def accel_ms2(val):
    """가속도 값을 m/s²로 변환"""
    return (twocomplements(val) / sensitive_accel) * 9.80665

def read_sensor_data():
    """IMU 센서에서 6축 데이터 읽기"""
    # 가속도 데이터
    accel_x = accel_ms2(read_data(register_accel_xout_h))
    accel_y = -accel_ms2(read_data(register_accel_yout_h))
    accel_z = accel_ms2(read_data(register_accel_zout_h))
    
    # 자이로스코프 데이터
    gyro_x = gyro_dps(read_data(register_gyro_xout_h))
    gyro_y = gyro_dps(read_data(register_gyro_yout_h))
    gyro_z = gyro_dps(read_data(register_gyro_zout_h))
    
    # 6축 데이터 반환 (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    return np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], dtype=np.float32)

# ========== 메인 실행 함수 ==========
def main():
    """Main execution function"""
    print("=" * 60)
    print("🚶 Real-time Gait Pattern Prediction System")
    print("=" * 60)
    print(f"⚙️  Configuration:")
    print(f"   - Sampling Rate: {SAMPLING_RATE}Hz")
    print(f"   - Window Size: {WINDOW_SIZE} frames (2 seconds)")
    print(f"   - Update: Every {STRIDE} frame (1/30 second)")
    print(f"   - Prediction Threshold: {THRESHOLD}")
    print(f"   - Model: {MODEL_PATH}")
    print(f"   - Scaler: {SCALER_PATH}")
    print()
    
    # Display options
    print("📊 Display Options:")
    print("   1. Prediction results only")
    print("   2. Sensor values only")
    print("   3. Both prediction results and sensor values")
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            if choice in ['1', '2', '3']:
                display_mode = int(choice)
                break
            else:
                print("❌ Please select 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nProgram terminated.")
            return
        except:
            print("❌ Please enter a valid number.")
    
    # Check file existence
    required_files = [MODEL_PATH, SCALER_PATH, ENCODER_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files and display_mode != 2:  # No model files needed for sensor-only mode
        print(f"❌ Required files missing: {missing_files}")
        print("Please ensure the following files exist in the current directory:")
        for file in required_files:
            print(f"   - {file}")
        return
    
    try:
        # Initialize sensor
        bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
        print("✅ IMU sensor initialized successfully")
        
        # Initialize predictor (except for sensor-only mode)
        predictor = None
        if display_mode != 2:
            predictor = RealTimeGaitPredictor(MODEL_PATH, SCALER_PATH, ENCODER_PATH, THRESHOLD)
        
        print(f"\n🚀 Starting real-time data collection (Mode {display_mode})")
        print("   Press Ctrl+C to stop")
        print()
        
        # Display headers
        if display_mode == 1:
            print("=" * 80)
            print(f"{'Time':<20} {'Predict':<10} {'Prob':<8} {'Conf':<8} {'Status'}")
            print("=" * 80)
        elif display_mode == 2:
            print("=" * 120)
            print(f"{'Time':<20} {'AccX':<8} {'AccY':<8} {'AccZ':<8} {'GyroX':<8} {'GyroY':<8} {'GyroZ':<8} {'Filtered Data'}")
            print("=" * 120)
        else:  # display_mode == 3
            print("=" * 140)
            print(f"{'Time':<20} {'AccX':<8} {'AccY':<8} {'AccZ':<8} {'GyroX':<8} {'GyroY':<8} {'GyroZ':<8} {'Predict':<8} {'Prob':<8} {'Status'}")
            print("=" * 140)
        
        # Timing control
        start_time = time.time()
        sample_count = 0
        target_interval = 1.0 / SAMPLING_RATE  # 30Hz = 0.0333 second interval
        
        prediction_count = 0
        gait_count = 0
        
        while True:
            current_time = time.time()
            timestamp_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Read sensor data
            sensor_data = read_sensor_data()
            
            # Process by mode
            if display_mode == 1:
                # Display prediction results only
                prediction_result = predictor.add_sensor_data(sensor_data)
                
                if prediction_result:
                    prediction_count += 1
                    
                    pred_class = prediction_result['predicted_class']
                    probability = prediction_result['probability']
                    confidence = prediction_result['confidence']
                    
                    if pred_class == 'gait':
                        gait_count += 1
                        status = "🚶 Walking"
                    else:
                        status = "🛑 Non-gait"
                    
                    print(f"{timestamp_str} {pred_class:<10} {probability:.4f}   {confidence:.4f}   {status}")
                    
                    # Show statistics every 10 seconds
                    if prediction_count % 300 == 0:
                        gait_ratio = gait_count / prediction_count * 100
                        print(f"\n📊 Statistics (Last 10 sec): Gait ratio {gait_ratio:.1f}% ({gait_count}/{prediction_count})")
                        print("=" * 80)
                        prediction_count = 0
                        gait_count = 0
            
            elif display_mode == 2:
                # Display sensor values only
                # Create temporary predictor to show filtered data
                if not hasattr(main, 'temp_predictor'):
                    main.temp_predictor = RealTimeGaitPredictor(MODEL_PATH, SCALER_PATH, ENCODER_PATH, THRESHOLD)
                
                filtered_data = main.temp_predictor.apply_filter(sensor_data)
                
                print(f"{timestamp_str} "
                      f"{sensor_data[0]:>7.3f} {sensor_data[1]:>7.3f} {sensor_data[2]:>7.3f} "
                      f"{sensor_data[3]:>7.2f} {sensor_data[4]:>7.2f} {sensor_data[5]:>7.2f} "
                      f"| {filtered_data[0]:>6.3f} {filtered_data[1]:>6.3f} {filtered_data[2]:>6.3f} "
                      f"{filtered_data[3]:>6.2f} {filtered_data[4]:>6.2f} {filtered_data[5]:>6.2f}")
            
            else:  # display_mode == 3
                # Display both prediction results and sensor values
                prediction_result = predictor.add_sensor_data(sensor_data)
                
                # Output sensor values every frame
                sensor_str = f"{sensor_data[0]:>7.3f} {sensor_data[1]:>7.3f} {sensor_data[2]:>7.3f} {sensor_data[3]:>7.2f} {sensor_data[4]:>7.2f} {sensor_data[5]:>7.2f}"
                
                if prediction_result:
                    prediction_count += 1
                    
                    pred_class = prediction_result['predicted_class']
                    probability = prediction_result['probability']
                    
                    if pred_class == 'gait':
                        gait_count += 1
                        status = "🚶 Walking"
                    else:
                        status = "🛑 Non-gait"
                    
                    print(f"{timestamp_str} {sensor_str} {pred_class:<8} {probability:.4f}   {status}")
                else:
                    # Output sensor values only when no prediction results
                    print(f"{timestamp_str} {sensor_str} {'Prep':<8} {'----'}     {'Collecting data'}")
            
            sample_count += 1
            
            # Timing adjustment (maintain 30Hz)
            elapsed = time.time() - start_time
            expected_time = sample_count * target_interval
            sleep_time = expected_time - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Real-time data collection stopped")
        
        # Save prediction results (only if predictor exists)
        if predictor:
            recent_predictions = predictor.get_recent_predictions(100)
            if recent_predictions:
                predictor.save_predictions()
                
                # Simple statistics
                total_predictions = len(recent_predictions)
                gait_predictions = sum(1 for p in recent_predictions if p['predicted_class'] == 'gait')
                gait_percentage = gait_predictions / total_predictions * 100
                
                print(f"\n📊 Session Statistics:")
                print(f"   Total predictions: {total_predictions}")
                print(f"   Gait predictions: {gait_predictions} ({gait_percentage:.1f}%)")
                print(f"   Non-gait predictions: {total_predictions - gait_predictions} ({100 - gait_percentage:.1f}%)")
    
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            bus.close()
            print("✅ I2C bus closed")
        except:
            pass

if __name__ == "__main__":
    main()