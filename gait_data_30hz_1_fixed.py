"""
보행 및 낙상 감지 IMU 센서 데이터 수집 프로그램 (100Hz 센서, 다운샘플링)
MODIFIED 2025-01-30: 낙상 감지용 100Hz 센서 수집, 보행 감지용 30Hz 다운샘플링
Features:
- 100Hz IMU 센서 데이터 수집 (멀티스레드)
- 실시간 낙상 감지 (TensorFlow Lite) - 100Hz 데이터 사용
- 실시간 보행 감지 (TensorFlow Lite) - 30Hz 다운샘플링 데이터 사용
- Supabase 직접 업로드
"""

from smbus2 import SMBus
from bitstring import Bits
import os
from dotenv import load_dotenv
import time
import datetime
import numpy as np
import threading
import pickle
from collections import deque
import tensorflow as tf
from supabase import create_client, Client
import io
import csv

# Global variables
bus = SMBus(1)
DEV_ADDR = 0x68

# IMU register addresses
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0

register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0

# Model paths
GAIT_MODEL_PATH = "models/gait_detection/model.tflite"
FALL_MODEL_PATH = "models/fall_detection/fall_detection.tflite"
GAIT_SCALER_PATH = "scalers/gait_detection"
FALL_SCALER_PATH = "scalers/fall_detection"

# Detection parameters
GAIT_WINDOW_SIZE = 60  # Window size for gait detection model
FALL_WINDOW_SIZE = 150  # Window size for fall detection model
SENSOR_HZ = 100  # 센서 데이터 수집 주파수 (100Hz)
GAIT_TARGET_HZ = 30   # 보행 감지용 다운샘플링 주파수 (30Hz)
GAIT_THRESHOLD = 0.24  # Gait detection threshold
FALL_THRESHOLD = 0.5  # Fall detection threshold

# State transition parameters
GAIT_TRANSITION_FRAMES = 60  # 2 seconds at 30Hz
MIN_GAIT_DURATION_FRAMES = 300  # 10 seconds at 30Hz

# Detection timing parameters - 목적별 최적화
FALL_DETECTION_INTERVAL = 0.05  # 낙상 감지 주기 (0.05초 = 20Hz) - 실시간성 강화
GAIT_DETECTION_INTERVAL = 0.1   # 보행 감지 주기 (0.1초 = 10Hz) - 정확도 우선, 배치 처리
GAIT_STRIDE = 1  # 보행 감지 stride (1 = 모든 프레임 처리)
GAIT_BATCH_SIZE = 5  # 보행 감지시 한번에 처리할 윈도우 수 (정확도 향상)

# Global Supabase client variable
supabase = None

# Global variables for sensor data collection
sensor_data_lock = threading.Lock()
raw_sensor_buffer = deque(maxlen=max(GAIT_WINDOW_SIZE * 4, FALL_WINDOW_SIZE * 2))  # 100Hz 버퍼 크기 조정
gait_downsampled_buffer = deque(maxlen=GAIT_WINDOW_SIZE * 3)  # 30Hz 다운샘플링된 데이터 버퍼
is_running = False

# Gait detection variables - 개선된 구조
gait_interpreter = None
gait_scaler = None
gait_label_encoder = None
gait_state = "non-gait"
gait_consecutive_count = 0  # 연속된 gait 예측 수
non_gait_consecutive_count = 0  # 연속된 non-gait 예측 수
current_gait_data = deque()  # deque로 변경하여 효율성 향상
current_gait_start_time = None
last_prediction_frame = -1  # 마지막 예측된 프레임 번호

# Fall detection variables
fall_interpreter = None
fall_scalers = {}  # Dictionary for multiple scalers

def read_data(register):
    """Read data from IMU register"""
    high = bus.read_byte_data(DEV_ADDR, register)
    low = bus.read_byte_data(DEV_ADDR, register+1)
    val = (high << 8) + low
    return val

def twocomplements(val):
    """Convert 2's complement"""
    s = Bits(uint=val, length=16)
    return s.int

def gyro_dps(val):
    """Convert gyroscope value to degrees/second"""
    return twocomplements(val)/sensitive_gyro

def accel_ms2(val):
    """Convert acceleration value to m/s²"""
    return (twocomplements(val)/sensitive_accel) * 9.80665

def init_supabase():
    """Initialize Supabase client with detailed debugging"""
    global supabase
    try:
        print("🔍 Supabase 초기화 시작...")
        
        # .env 파일 존재 확인
        env_path = ".env"
        if os.path.exists(env_path):
            print(f"✅ .env 파일 발견: {os.path.abspath(env_path)}")
        else:
            print(f"⚠️ .env 파일이 현재 디렉토리에 없습니다: {os.path.abspath(env_path)}")
            print(f"   현재 작업 디렉토리: {os.getcwd()}")
            print(f"   디렉토리 내용: {os.listdir('.')}")
        
        # 환경변수 로드
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        # 환경변수 디버깅 정보
        print(f"🔗 SUPABASE_URL: {'✅ 로드됨' if url else '❌ 없음'}")
        if url:
            print(f"   URL: {url}")
        print(f"🔑 SUPABASE_KEY: {'✅ 로드됨' if key else '❌ 없음'}")
        if key:
            print(f"   Key prefix: {key[:20]}...")
        
        if not url or not key:
            print("❌ 환경변수가 올바르게 로드되지 않았습니다.")
            print("   다음을 확인해주세요:")
            print("   1. .env 파일이 스크립트와 같은 디렉토리에 있는지")
            print("   2. .env 파일에 SUPABASE_URL=your_url 형식으로 작성되어 있는지")
            print("   3. .env 파일에 SUPABASE_KEY=your_key 형식으로 작성되어 있는지")
            raise RuntimeError("Supabase 환경변수 누락 (SUPABASE_URL 또는 SUPABASE_KEY)")
        
        print("🔄 Supabase 클라이언트 생성 중...")
        supabase = create_client(url, key)
        print("✅ Supabase 클라이언트 생성 성공")
        
        # 네트워크 연결 테스트
        print("🌐 네트워크 연결 테스트 중...")
        import requests
        try:
            response = requests.get(f"{url}/rest/v1/", headers={"apikey": key}, timeout=10)
            print(f"✅ 네트워크 연결 성공 (HTTP {response.status_code})")
        except requests.exceptions.ConnectionError:
            print("❌ 네트워크 연결 실패 - 인터넷 연결을 확인해주세요")
            return False
        except requests.exceptions.Timeout:
            print("❌ 연결 시간 초과 - 네트워크가 느리거나 불안정합니다")
            return False
        except Exception as net_error:
            print(f"⚠️ 네트워크 테스트 중 오류: {net_error}")
        
        # Supabase 인증 테스트
        print("🔐 Supabase 인증 테스트 중...")
        try:
            # 간단한 테스트 쿼리 실행
            response = supabase.table('fall_history').select('count').limit(1).execute()
            print("✅ Supabase 인증 성공!")
            print("✅ Supabase 초기화 완료")
            return True
        except Exception as auth_error:
            print(f"❌ Supabase 인증 실패: {auth_error}")
            print("   다음을 확인해주세요:")
            print("   1. API 키가 올바른지")
            print("   2. 프로젝트 URL이 정확한지")
            print("   3. fall_history 테이블이 존재하는지")
            print("   4. API 키에 해당 테이블 접근 권한이 있는지")
            return False
            
    except Exception as e:
        print(f"❌ Supabase 초기화 실패: {e}")
        print(f"   오류 타입: {type(e).__name__}")
        return False

def load_models():
    """Load gait and fall detection models with version compatibility"""
    global gait_interpreter, gait_scaler, gait_label_encoder, fall_interpreter, fall_scalers
    
    # Check scikit-learn version
    try:
        import sklearn
        print(f"📦 scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not installed")
    
    # Load gait detection model
    try:
        if os.path.exists(GAIT_MODEL_PATH):
            gait_interpreter = tf.lite.Interpreter(model_path=GAIT_MODEL_PATH)
            gait_interpreter.allocate_tensors()
            print(f"✅ Gait model loaded: {GAIT_MODEL_PATH}")
        
        # Load gait scaler with version compatibility
        gait_scaler_file = os.path.join(GAIT_SCALER_PATH, "standard_scaler.pkl")
        if os.path.exists(gait_scaler_file):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    with open(gait_scaler_file, 'rb') as f:
                        gait_scaler = pickle.load(f)
                print(f"✅ Gait StandardScaler loaded (version compatibility handled)")
            except Exception as scaler_error:
                print(f"⚠️ Gait StandardScaler loading failed: {scaler_error}")
                print("   Continuing without gait scaler - manual scaling may be needed")
                gait_scaler = None
        
        # Load gait label encoder with version compatibility
        gait_label_encoder_file = os.path.join(GAIT_SCALER_PATH, "label_encoder.pkl")
        if os.path.exists(gait_label_encoder_file):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    with open(gait_label_encoder_file, 'rb') as f:
                        gait_label_encoder = pickle.load(f)
                print(f"✅ Gait LabelEncoder loaded (version compatibility handled)")
            except Exception as encoder_error:
                print(f"⚠️ Gait LabelEncoder loading failed: {encoder_error}")
                print("   Continuing without gait label encoder - prediction results will be numeric")
                gait_label_encoder = None
        else:
            print(f"⚠️ Gait LabelEncoder file not found: {gait_label_encoder_file}")
            gait_label_encoder = None
            
    except Exception as e:
        print(f"❌ Gait model loading error: {e}")
    
    # Load fall detection model
    try:
        if os.path.exists(FALL_MODEL_PATH):
            fall_interpreter = tf.lite.Interpreter(model_path=FALL_MODEL_PATH)
            fall_interpreter.allocate_tensors()
            print(f"✅ Fall model loaded: {FALL_MODEL_PATH}")
        
        # Load fall scalers (both minmax and standard) with version compatibility
        scalers_loaded = 0
        total_scalers = 0
        
        for sensor in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
            minmax_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_minmax_scaler.pkl")
            standard_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_standard_scaler.pkl")
            
            # Load MinMax scaler
            if os.path.exists(minmax_file):
                total_scalers += 1
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        with open(minmax_file, 'rb') as f:
                            fall_scalers[f"{sensor}_minmax"] = pickle.load(f)
                    scalers_loaded += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {sensor}_minmax scaler: {e}")
            
            # Load Standard scaler
            if os.path.exists(standard_file):
                total_scalers += 1
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        with open(standard_file, 'rb') as f:
                            fall_scalers[f"{sensor}_standard"] = pickle.load(f)
                    scalers_loaded += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {sensor}_standard scaler: {e}")
        
        print(f"✅ Fall scalers loaded: {scalers_loaded}/{total_scalers}")
        
        if scalers_loaded == 0:
            print("⚠️ No fall scalers loaded - fall detection may not work correctly")
        elif scalers_loaded < total_scalers:
            print("⚠️ Some fall scalers failed to load - fall detection accuracy may be reduced")
            
    except Exception as e:
        print(f"❌ Fall model loading error: {e}")

def sensor_collection_thread():
    """Thread for collecting sensor data at 100Hz"""
    global raw_sensor_buffer, gait_downsampled_buffer, is_running
    
    start_time = time.time()
    frame_count = 0
    gait_frame_count = 0
    last_gait_sample_time = 0
    gait_sampling_interval = 1.0 / GAIT_TARGET_HZ  # 30Hz를 위한 샘플링 간격
    
    while is_running:
        try:
            # Read IMU sensor data
            accel_x = accel_ms2(read_data(register_accel_xout_h))
            accel_y = -accel_ms2(read_data(register_accel_yout_h))
            accel_z = accel_ms2(read_data(register_accel_zout_h))
            
            gyro_x = gyro_dps(read_data(register_gyro_xout_h))
            gyro_y = gyro_dps(read_data(register_gyro_yout_h))
            gyro_z = gyro_dps(read_data(register_gyro_zout_h))
            
            # Calculate timestamp
            current_time = time.time()
            sync_timestamp = current_time - start_time
            
            # Store raw sensor data (100Hz) for fall detection
            sensor_data = {
                'frame': frame_count,
                'sync_timestamp': sync_timestamp,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'unix_timestamp': current_time
            }
            
            with sensor_data_lock:
                # 100Hz 버퍼에 추가 (낙상 감지용)
                if len(raw_sensor_buffer) >= raw_sensor_buffer.maxlen:
                    raw_sensor_buffer.popleft()
                raw_sensor_buffer.append(sensor_data)
                
                # 30Hz 다운샘플링 (보행 감지용)
                if sync_timestamp - last_gait_sample_time >= gait_sampling_interval:
                    gait_sensor_data = sensor_data.copy()
                    gait_sensor_data['gait_frame'] = gait_frame_count
                    
                    if len(gait_downsampled_buffer) >= gait_downsampled_buffer.maxlen:
                        gait_downsampled_buffer.popleft()
                    gait_downsampled_buffer.append(gait_sensor_data)
                    
                    last_gait_sample_time = sync_timestamp
                    gait_frame_count += 1
            
            frame_count += 1
            
            # Maintain 100Hz sampling rate
            next_sample_time = start_time + (frame_count * (1.0 / SENSOR_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"❌ Sensor collection error: {e}")
            time.sleep(0.001)

def preprocess_for_gait(sensor_window):
    """Preprocess sensor data for gait detection"""
    if not gait_scaler:
        return None
    
    try:
        # Extract sensor values in correct order
        sensor_array = np.array([[
            data['accel_x'], data['accel_y'], data['accel_z'],
            data['gyro_x'], data['gyro_y'], data['gyro_z']
        ] for data in sensor_window], dtype=np.float32)
        
        # Reshape for scaler
        sensor_array = sensor_array.reshape(1, GAIT_WINDOW_SIZE, 6)
        n_samples, n_frames, n_features = sensor_array.shape
        sensor_reshaped = sensor_array.reshape(-1, n_features)
        
        # Apply Standard scaling
        sensor_scaled = gait_scaler.transform(sensor_reshaped)
        sensor_scaled = sensor_scaled.reshape(n_samples, n_frames, n_features).astype(np.float32)
        
        return sensor_scaled
    except Exception as e:
        print(f"❌ Gait preprocessing error: {e}")
        return None

def preprocess_for_fall(sensor_window):
    """Preprocess sensor data for fall detection"""
    if not fall_scalers:
        return None
    
    try:
        # Process each sensor channel
        processed_data = []
        
        for data in sensor_window:
            # Apply transformations for fall detection
            acc_x = data['accel_x'] / 9.80665
            acc_y = data['accel_y'] / 9.80665  # Sign change
            acc_z = data['accel_z'] / 9.80665
            gyr_x = data['gyro_x']
            gyr_y = data['gyro_y']
            gyr_z = data['gyro_z']
            
            processed_data.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z])
        
        # Convert to numpy array
        sensor_array = np.array(processed_data, dtype=np.float32)
        
        # Apply scalers for each sensor channel
        sensor_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        scaled_data = np.zeros_like(sensor_array)
        
        for i, sensor_name in enumerate(sensor_names):
            # Apply MinMax scaler first if available
            if f"{sensor_name}_minmax" in fall_scalers:
                scaled_data[:, i] = fall_scalers[f"{sensor_name}_minmax"].transform(
                    sensor_array[:, i].reshape(-1, 1)
                ).flatten()
            else:
                scaled_data[:, i] = sensor_array[:, i]
            
            # Apply Standard scaler after MinMax if available
            if f"{sensor_name}_standard" in fall_scalers:
                scaled_data[:, i] = fall_scalers[f"{sensor_name}_standard"].transform(
                    scaled_data[:, i].reshape(-1, 1)
                ).flatten()
        
        return scaled_data.reshape(1, FALL_WINDOW_SIZE, 6)
    except Exception as e:
        print(f"❌ Fall preprocessing error: {e}")
        return None

def gait_detection_thread():
    """Thread for gait detection using 30Hz downsampled data with batch processing for accuracy"""
    global gait_state, gait_consecutive_count, non_gait_consecutive_count
    global current_gait_data, current_gait_start_time, last_prediction_frame
    
    print("🚶 Gait detection thread initialized (30Hz downsampled, batch processing for accuracy)")
    
    while is_running:
        try:
            # Get available downsampled sensor data (30Hz)
            with sensor_data_lock:
                if len(gait_downsampled_buffer) < GAIT_WINDOW_SIZE + GAIT_BATCH_SIZE:
                    time.sleep(0.01)
                    continue
                
                # 배치 처리를 위해 여러 윈도우 준비
                buffer_list = list(gait_downsampled_buffer)
                available_frames = len(buffer_list)
            
            # 배치 처리: stride=1로 여러 윈도우 생성
            windows_to_process = []
            frame_numbers = []
            
            # 마지막 처리된 프레임 이후부터 처리
            start_idx = max(0, available_frames - GAIT_WINDOW_SIZE - GAIT_BATCH_SIZE + 1)
            
            for i in range(GAIT_BATCH_SIZE):
                window_start = start_idx + i * GAIT_STRIDE
                window_end = window_start + GAIT_WINDOW_SIZE
                
                if window_end <= available_frames:
                    window = buffer_list[window_start:window_end]
                    current_frame = window[-1]['gait_frame']
                    
                    # 이미 처리된 프레임은 건너뛰기
                    if current_frame > last_prediction_frame:
                        windows_to_process.append(window)
                        frame_numbers.append(current_frame)
            
            # 배치로 예측 수행
            if windows_to_process and gait_interpreter and gait_scaler:
                batch_predictions = []
                
                for window in windows_to_process:
                    preprocessed = preprocess_for_gait(window)
                    if preprocessed is not None:
                        # Run inference
                        input_details = gait_interpreter.get_input_details()
                        output_details = gait_interpreter.get_output_details()
                        
                        gait_interpreter.set_tensor(input_details[0]['index'], preprocessed)
                        gait_interpreter.invoke()
                        
                        prediction = gait_interpreter.get_tensor(output_details[0]['index'])
                        gait_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
                        batch_predictions.append((gait_probability, window[-1]))
                
                # 배치 결과를 평균내어 더 안정적인 예측 (정확도 향상)
                if batch_predictions:
                    avg_probability = np.mean([pred[0] for pred in batch_predictions])
                    latest_sensor_data = batch_predictions[-1][1]  # 가장 최신 데이터 사용
                    
                    # Label decoding for better debugging
                    if gait_label_encoder:
                        binary_pred = 1 if avg_probability > GAIT_THRESHOLD else 0
                        try:
                            predicted_label = gait_label_encoder.inverse_transform([binary_pred])[0]
                            # Debug information with label
                            current_frame = latest_sensor_data['gait_frame']
                            if current_frame % 30 == 0:  # Print every 1 second at 30Hz
                                print(f"🔍 Frame {current_frame}: Batch Avg Prob={avg_probability:.3f}, Pred={predicted_label} (batch size: {len(batch_predictions)})")
                        except Exception as e:
                            print(f"⚠️ Label decoding failed: {e}")
                    
                    # Update consecutive counts using averaged probability
                    if avg_probability > GAIT_THRESHOLD:
                        gait_consecutive_count += len(batch_predictions)  # 배치 크기만큼 증가
                        non_gait_consecutive_count = 0
                    else:
                        non_gait_consecutive_count += len(batch_predictions)  # 배치 크기만큼 증가
                        gait_consecutive_count = 0
                    
                    # State transition logic
                    update_gait_state_accurate(latest_sensor_data, avg_probability, len(batch_predictions))
                    
                    # 마지막 처리된 프레임 업데이트
                    last_prediction_frame = frame_numbers[-1]
            
            time.sleep(GAIT_DETECTION_INTERVAL)  # 정확도 우선 보행 감지 주기
            
        except Exception as e:
            print(f"❌ Gait detection error: {e}")
            time.sleep(0.1)

def update_gait_state_accurate(latest_sensor_data, avg_probability, batch_size):
    """정확도 향상을 위한 배치 기반 보행 상태 업데이트"""
    global gait_state, current_gait_data, current_gait_start_time
    
    if gait_state == "non-gait":
        # Gait 시작 조건: 배치 평균 확률 기반으로 더 안정적인 판단
        adjusted_threshold_frames = GAIT_TRANSITION_FRAMES // batch_size  # 배치 크기 고려한 임계값 조정
        if gait_consecutive_count >= adjusted_threshold_frames:
            gait_state = "gait"
            current_gait_start_time = latest_sensor_data['unix_timestamp']
            current_gait_data = deque()  # 새로운 gait 데이터 시작
            
            # Show label-decoded result
            label_info = ""
            if gait_label_encoder:
                try:
                    predicted_label = gait_label_encoder.inverse_transform([1])[0]  # 1 = gait
                    label_info = f" -> {predicted_label}"
                except:
                    pass
            
            print(f"🚶 Gait started at gait_frame {latest_sensor_data['gait_frame']} (avg prob: {avg_probability:.3f}, batch confidence: {gait_consecutive_count}/{adjusted_threshold_frames}){label_info}")
    
    elif gait_state == "gait":
        # 보행 중인 경우 데이터 수집 (배치의 모든 프레임 추가하지 않고 최신 데이터만)
        current_gait_data.append(latest_sensor_data)
        
        # Gait 종료 조건: 배치 평균 확률 기반으로 더 안정적인 판단
        adjusted_threshold_frames = GAIT_TRANSITION_FRAMES // batch_size  # 배치 크기 고려한 임계값 조정
        if non_gait_consecutive_count >= adjusted_threshold_frames:
            # 보행 데이터 저장 체크
            gait_duration_frames = len(current_gait_data)
            gait_duration_seconds = gait_duration_frames / GAIT_TARGET_HZ
            
            # Show label-decoded result
            label_info = ""
            if gait_label_encoder:
                try:
                    predicted_label = gait_label_encoder.inverse_transform([0])[0]  # 0 = non-gait
                    label_info = f" -> {predicted_label}"
                except:
                    pass
            
            print(f"🛑 Gait ended at gait_frame {latest_sensor_data['gait_frame']} (avg prob: {avg_probability:.3f}, duration: {gait_duration_frames} frames, {gait_duration_seconds:.1f}s){label_info}")
            
            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(list(current_gait_data))
                print(f"✅ Gait data saved ({gait_duration_frames} frames)")
            else:
                print(f"⚠️ Gait duration too short: {gait_duration_frames} frames ({gait_duration_seconds:.1f}s < {MIN_GAIT_DURATION_FRAMES/GAIT_TARGET_HZ:.1f}s)")
            
            # 상태 리셋
            gait_state = "non-gait"
            current_gait_data = deque()
            current_gait_start_time = None

def update_gait_state_simple(latest_sensor_data, gait_probability):
    """간단하고 효율적인 보행 상태 업데이트 (30Hz 기준)"""
    global gait_state, current_gait_data, current_gait_start_time
    
    if gait_state == "non-gait":
        # Gait 시작 조건: 연속으로 GAIT_TRANSITION_FRAMES 만큼 gait로 예측됨
        if gait_consecutive_count >= GAIT_TRANSITION_FRAMES:
            gait_state = "gait"
            current_gait_start_time = latest_sensor_data['unix_timestamp']
            current_gait_data = deque()  # 새로운 gait 데이터 시작
            
            # Show label-decoded result
            label_info = ""
            if gait_label_encoder:
                try:
                    predicted_label = gait_label_encoder.inverse_transform([1])[0]  # 1 = gait
                    label_info = f" -> {predicted_label}"
                except:
                    pass
            
            print(f"🚶 Gait started at gait_frame {latest_sensor_data['gait_frame']} (confidence: {gait_consecutive_count}/{GAIT_TRANSITION_FRAMES}){label_info}")
    
    elif gait_state == "gait":
        # 보행 중인 경우 데이터 수집
        current_gait_data.append(latest_sensor_data)
        
        # Gait 종료 조건: 연속으로 GAIT_TRANSITION_FRAMES 만큼 non-gait로 예측됨
        if non_gait_consecutive_count >= GAIT_TRANSITION_FRAMES:
            # 보행 데이터 저장 체크
            gait_duration_frames = len(current_gait_data)
            gait_duration_seconds = gait_duration_frames / GAIT_TARGET_HZ
            
            # Show label-decoded result
            label_info = ""
            if gait_label_encoder:
                try:
                    predicted_label = gait_label_encoder.inverse_transform([0])[0]  # 0 = non-gait
                    label_info = f" -> {predicted_label}"
                except:
                    pass
            
            print(f"🛑 Gait ended at gait_frame {latest_sensor_data['gait_frame']} (duration: {gait_duration_frames} frames, {gait_duration_seconds:.1f}s){label_info}")
            
            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(list(current_gait_data))
                print(f"✅ Gait data saved ({gait_duration_frames} frames)")
            else:
                print(f"⚠️ Gait duration too short: {gait_duration_frames} frames ({gait_duration_seconds:.1f}s < {MIN_GAIT_DURATION_FRAMES/GAIT_TARGET_HZ:.1f}s)")
            
            # 상태 리셋
            gait_state = "non-gait"
            current_gait_data = deque()
            current_gait_start_time = None

def fall_detection_thread():
    """Thread for fall detection using 100Hz data with longer interval"""
    print("🚨 Fall detection thread initialized (100Hz data, 0.2s interval)")
    
    while is_running:
        try:
            with sensor_data_lock:
                if len(raw_sensor_buffer) >= FALL_WINDOW_SIZE:
                    sensor_window = list(raw_sensor_buffer)[-FALL_WINDOW_SIZE:]
                else:
                    time.sleep(0.1)
                    continue
            
            # Preprocess and predict
            if fall_interpreter and fall_scalers:
                preprocessed = preprocess_for_fall(sensor_window)
                if preprocessed is not None:
                    # Run inference
                    input_details = fall_interpreter.get_input_details()
                    output_details = fall_interpreter.get_output_details()
                    
                    fall_interpreter.set_tensor(input_details[0]['index'], preprocessed)
                    fall_interpreter.invoke()
                    
                    prediction = fall_interpreter.get_tensor(output_details[0]['index'])
                    fall_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
                    
                    # Check for fall
                    if fall_probability > FALL_THRESHOLD:
                        print(f"🚨 Fall detected! Probability: {fall_probability:.2f}")
                        save_fall_event_to_supabase(sensor_window[-1]['unix_timestamp'])
            
            time.sleep(FALL_DETECTION_INTERVAL)  # 더 긴 간격으로 낙상 감지 (0.2초 = 5Hz)
            
        except Exception as e:
            print(f"❌ Fall detection error: {e}")
            time.sleep(0.1)

def save_gait_data_to_supabase(gait_data):
    """Save gait data as CSV to Supabase (30Hz downsampled data)"""
    if not supabase:
        print("❌ Supabase not initialized")
        return
    
    try:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['gait_frame', 'sync_timestamp', 'accel_x', 'accel_y', 'accel_z', 
                        'gyro_x', 'gyro_y', 'gyro_z'])
        
        # Write data with proper formatting
        first_timestamp = gait_data[0]['sync_timestamp']
        for data in gait_data:
            writer.writerow([
                data['gait_frame'],
                f"{data['sync_timestamp'] - first_timestamp:.6f}",  # Relative timestamp from 0
                f"{data['accel_x']:.3f}",
                f"{data['accel_y']:.3f}",
                f"{data['accel_z']:.3f}",
                f"{data['gyro_x']:.5f}",
                f"{data['gyro_y']:.5f}",
                f"{data['gyro_z']:.5f}"
            ])
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gait_data_30hz_{timestamp}.csv"
        
        # Convert to bytes
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode('utf-8')
        
        # Upload to Supabase Storage with retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = supabase.storage.from_('gait-data').upload(
                    file=csv_bytes,
                    path=filename,
                    file_options={"content-type": "text/csv"}
                )
                print(f"✅ Gait data saved: {filename} ({len(gait_data)} frames)")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Upload attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
        
    except Exception as e:
        print(f"❌ Failed to save gait data: {e}")
        # Save to local file as backup
        try:
            backup_filename = f"backup_{filename}"
            with open(backup_filename, 'w') as f:
                f.write(csv_content)
            print(f"✅ Backup saved to local file: {backup_filename}")
        except Exception as backup_e:
            print(f"❌ Failed to save backup: {backup_e}")

def save_fall_event_to_supabase(timestamp):
    """Save fall event to Supabase database"""
    if not supabase:
        print("❌ Supabase not initialized")
        return
    
    try:
        # Insert fall event into Fall History table
        data = {
            'timestamp': datetime.datetime.fromtimestamp(timestamp).isoformat(),
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        response = supabase.table('fall_history').insert(data).execute()
        print(f"✅ Fall event saved to database")
        
    except Exception as e:
        print(f"❌ Failed to save fall event: {e}")

def main():
    """Main execution function"""
    global is_running
    
    print("=" * 70)
    print("🚶 Gait & Fall Detection System (Optimized for Different Priorities)")
    print("=" * 70)
    print(f"📊 Sensor collection: {SENSOR_HZ}Hz")
    print(f"🚶 Gait detection: {GAIT_TARGET_HZ}Hz (accuracy priority - batch processing)")
    print(f"   └─ Batch size: {GAIT_BATCH_SIZE}, Stride: {GAIT_STRIDE}, Interval: {GAIT_DETECTION_INTERVAL}s")
    print(f"🚨 Fall detection: {SENSOR_HZ}Hz (real-time priority)")
    print(f"   └─ Interval: {FALL_DETECTION_INTERVAL}s ({1/FALL_DETECTION_INTERVAL:.0f}Hz detection)")
    print("=" * 70)
    
    # Initialize Supabase
    if not init_supabase():
        print("⚠️ Continuing without Supabase - data will be saved locally only")
        print("⚠️ Please check your SUPABASE_URL and SUPABASE_KEY in .env file")
    
    # Load models
    load_models()
    
    # Initialize IMU sensor
    bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
    print("✅ IMU sensor initialized")
    
    # Start threads
    is_running = True
    
    sensor_thread = threading.Thread(target=sensor_collection_thread)
    sensor_thread.daemon = True
    sensor_thread.start()
    print("✅ Sensor collection thread started (100Hz)")
    
    gait_thread = threading.Thread(target=gait_detection_thread)
    gait_thread.daemon = True
    gait_thread.start()
    print("✅ Gait detection thread started (30Hz downsampled)")
    
    fall_thread = threading.Thread(target=fall_detection_thread)
    fall_thread.daemon = True
    fall_thread.start()
    print("✅ Fall detection thread started (100Hz, 0.2s interval)")
    
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Print status every 5 seconds
            if int(time.time()) % 5 == 0:
                with sensor_data_lock:
                    raw_buffer_size = len(raw_sensor_buffer)
                    gait_buffer_size = len(gait_downsampled_buffer)
                
                gait_data_size = len(current_gait_data) if current_gait_data else 0
                
                print(f"📊 Status - Raw(100Hz): {raw_buffer_size}, Gait(30Hz): {gait_buffer_size}, "
                      f"State: {gait_state}, Gait count: +{gait_consecutive_count}/-{non_gait_consecutive_count}, "
                      f"Gait frames: {gait_data_size}, Last gait frame: {last_prediction_frame}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        
    finally:
        is_running = False
        time.sleep(1)  # Allow threads to finish
        
        # Save any remaining gait data
        if gait_state == "gait" and current_gait_data:
            gait_duration_frames = len(current_gait_data)
            print(f"💾 Saving remaining gait data: {gait_duration_frames} frames")
            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(list(current_gait_data))
                print(f"✅ Final gait data saved ({gait_duration_frames} frames)")
            else:
                print(f"⚠️ Final gait data too short: {gait_duration_frames} frames")
        
        bus.close()
        print("✅ System stopped")

if __name__ == "__main__":
    main() 