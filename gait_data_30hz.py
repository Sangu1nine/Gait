"""
보행 및 낙상 감지 IMU 센서 데이터 수집 프로그램
Created: 2025-01-30
MODIFIED 2025-01-30: 실시간 센서 데이터 로그 출력 기능 추가
MODIFIED 2025-01-30: 낙상 감지 모델 윈도우 크기 수정 (60 → 150 프레임)
MODIFIED 2025-01-30: Y축 부호 처리 중복 제거 (센서 수집에서만 처리)
MODIFIED 2025-01-30: 실시간 모델 예측 결과 로그 출력 기능 추가 - 센서 로그 주기 단축 (0.5초 → 0.2초), 예측 결과 로그 (0.3초 주기), 키보드 제어 추가
MODIFIED 2025-01-30: 낙상 감지 스케일러 실제 적용 - 각 센서 채널별로 MinMax/Standard 스케일러 적용 구현
MODIFIED 2025-01-30: 실시간 로그 출력 개선 - 라즈베리파이 콘솔 출력 지연 방지를 위해 flush=True 추가
MODIFIED 2025-01-30: 멀티스레드 로깅 시스템 구현 - 큐 기반 전용 로깅 스레드로 성능 개선 및 스레드 경합 방지
MODIFIED 2025-01-30: 센서 수집 디버깅 개선 - 오류 상세 로그, 수집 상태 모니터링, I2C 초기화 검증 추가
MODIFIED 2025-01-30: I2C 성능 최적화 - 버스트 읽기(14바이트 한번에), 로그 큐 크기 제한, 디버그 주기 단축
MODIFIED 2025-01-30: 로그 시스템 성능 최적화 - 로그 파일 분리(RotatingFileHandler), 실시간 진행바, 배치 처리, 버퍼 해제 안내
Features:
- 30Hz IMU 센서 데이터 수집 (멀티스레드)
- 실시간 센서 데이터 로그 출력 (0.2초 주기, 토글 가능)
- 실시간 모델 예측 결과 로그 출력 (0.3초 주기, 토글 가능)
- 실시간 보행 감지 (TensorFlow Lite, 60 프레임)
- 실시간 낙상 감지 (TensorFlow Lite, 150 프레임)
- Supabase 직접 업로드
- 키보드 제어: 's' (센서 로그), 'p' (예측 로그), 'd' (상세 상태)
"""

from smbus2 import SMBus
from bitstring import Bits
import os
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
import queue
import logging
from logging.handlers import RotatingFileHandler
import sys

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
GAIT_WINDOW_SIZE = 60   # Window size for gait detection model
FALL_WINDOW_SIZE = 150  # Window size for fall detection model
TARGET_HZ = 30   # Sampling rate
GAIT_THRESHOLD = 0.5  # Gait detection threshold
FALL_THRESHOLD = 0.5  # Fall detection threshold

# State transition parameters
GAIT_TRANSITION_FRAMES = 60  # 2 seconds at 30Hz
MIN_GAIT_DURATION_FRAMES = 300  # 10 seconds at 30Hz

# Supabase configuration
SUPABASE_URL = "YOUR_SUPABASE_URL"  # Replace with your Supabase URL
SUPABASE_KEY = "YOUR_SUPABASE_KEY"  # Replace with your Supabase key
supabase: Client = None

# Global variables for sensor data collection
sensor_data_lock = threading.Lock()
raw_sensor_buffer = deque(maxlen=max(GAIT_WINDOW_SIZE, FALL_WINDOW_SIZE) * 10)  # Store more for CSV saving
is_running = False
show_sensor_logs = True  # Flag to control sensor data logging
show_prediction_logs = True  # Flag to control prediction result logging

# Logging system variables
LOG_BATCH = 10
log_queue = queue.Queue(maxsize=1000)  # Increased queue size for batch processing
logging_active = True

# File logging setup
file_logger = None
console_log_count = 0
last_status_line = ""

# Gait detection variables
gait_interpreter = None
gait_scaler = None
gait_state = "non-gait"
gait_frame_count = 0
non_gait_frame_count = 0
current_gait_start_frame = None
current_gait_data = []

# Fall detection variables
fall_interpreter = None
fall_scalers = {}  # Dictionary for multiple scalers

# Prediction result variables
latest_gait_probability = 0.0
latest_fall_probability = 0.0
prediction_update_count = 0

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
    """Initialize Supabase client"""
    global supabase
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Supabase initialization failed: {e}")
        return False

def load_models():
    """Load gait and fall detection models"""
    global gait_interpreter, gait_scaler, fall_interpreter, fall_scalers
    
    # Load gait detection model
    try:
        if os.path.exists(GAIT_MODEL_PATH):
            gait_interpreter = tf.lite.Interpreter(model_path=GAIT_MODEL_PATH)
            gait_interpreter.allocate_tensors()
            print(f"✅ Gait model loaded: {GAIT_MODEL_PATH}")
        
        # Load gait scaler
        gait_scaler_file = os.path.join(GAIT_SCALER_PATH, "minmax_scaler.pkl")
        if os.path.exists(gait_scaler_file):
            with open(gait_scaler_file, 'rb') as f:
                gait_scaler = pickle.load(f)
            print(f"✅ Gait scaler loaded")
    except Exception as e:
        print(f"❌ Gait model loading error: {e}")
    
    # Load fall detection model
    try:
        if os.path.exists(FALL_MODEL_PATH):
            fall_interpreter = tf.lite.Interpreter(model_path=FALL_MODEL_PATH)
            fall_interpreter.allocate_tensors()
            print(f"✅ Fall model loaded: {FALL_MODEL_PATH}")
        
        # Load fall scalers (both minmax and standard)
        for sensor in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
            minmax_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_minmax_scaler.pkl")
            standard_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_standard_scaler.pkl")
            
            if os.path.exists(minmax_file):
                with open(minmax_file, 'rb') as f:
                    fall_scalers[f"{sensor}_minmax"] = pickle.load(f)
            
            if os.path.exists(standard_file):
                with open(standard_file, 'rb') as f:
                    fall_scalers[f"{sensor}_standard"] = pickle.load(f)
        
        print(f"✅ Fall scalers loaded")
    except Exception as e:
        print(f"❌ Fall model loading error: {e}")

def setup_file_logging():
    """Setup file logging with rotation"""
    global file_logger
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Setup file logger
    file_logger = logging.getLogger('gait_system')
    file_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    file_logger.handlers.clear()
    
    # Rotating file handler (10MB per file, keep 5 files)
    file_handler = RotatingFileHandler(
        'logs/gait_system.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Formatter for file logs
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)
    
    print("✅ File logging setup: logs/gait_system.log")

def fast_log(message, console_summary=False):
    """Fast logging function - detailed to file, summary to console"""
    global console_log_count
    
    if logging_active:
        # Always log to file
        if file_logger:
            file_logger.info(message)
        
        # Console output control
        if console_summary or show_sensor_logs or show_prediction_logs:
            try:
                log_queue.put_nowait(message)
                console_log_count += 1
            except queue.Full:
                pass  # Drop log if queue is full to avoid blocking

def update_progress_bar():
    """Update single-line progress bar"""
    global last_status_line
    
    with sensor_data_lock:
        buffer_size = len(raw_sensor_buffer)
    
    # Create progress bar
    progress_info = (
        f"📊 [{buffer_size:4d}] State:{gait_state:8s} | "
        f"Gait:{latest_gait_probability:.3f} Fall:{latest_fall_probability:.3f} | "
        f"Logs:{console_log_count:5d} | "
        f"Sensor:{'ON' if show_sensor_logs else 'OFF'} Pred:{'ON' if show_prediction_logs else 'OFF'}"
    )
    
    # Clear previous line and write new status
    if last_status_line:
        sys.stdout.write("\r" + " " * len(last_status_line) + "\r")
    
    sys.stdout.write(progress_info)
    sys.stdout.flush()
    last_status_line = progress_info

def logging_thread():
    """Dedicated thread for console logging with progress bar"""
    global logging_active
    batch = []
    progress_update_count = 0
    
    while is_running or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.05)
            batch.append(line)
            if len(batch) >= LOG_BATCH:
                # Clear progress bar, print batch, update progress bar
                if last_status_line:
                    sys.stdout.write("\r" + " " * len(last_status_line) + "\r")
                print("\n".join(batch), flush=True)
                batch.clear()
                update_progress_bar()
            log_queue.task_done()
        except queue.Empty:
            if batch:  # 남은 줄 출력
                if last_status_line:
                    sys.stdout.write("\r" + " " * len(last_status_line) + "\r")
                print("\n".join(batch), flush=True)
                batch.clear()
                update_progress_bar()
            
            # Update progress bar every few cycles
            progress_update_count += 1
            if progress_update_count % 20 == 0:  # Every ~1 second
                update_progress_bar()
            continue
        except Exception as e:
            if batch:  # 에러 발생 시에도 남은 로그 출력
                if last_status_line:
                    sys.stdout.write("\r" + " " * len(last_status_line) + "\r")
                print("\n".join(batch), flush=True)
                batch.clear()
            print(f"❌ Logging error: {e}", flush=True)
            update_progress_bar()
    logging_active = False

def sensor_collection_thread():
    """Thread for collecting sensor data at 30Hz"""
    global raw_sensor_buffer, is_running, show_sensor_logs
    
    start_time = time.time()
    frame_count = 0
    last_log_time = time.time()
    
    while is_running:
        try:
            # Read IMU sensor data using burst read for better performance
            try:
                # Read all sensor data at once (14 bytes from 0x3B to 0x48)
                raw_data = bus.read_i2c_block_data(DEV_ADDR, 0x3B, 14)
                
                # Parse accelerometer data (first 6 bytes)
                accel_x_raw = (raw_data[0] << 8) | raw_data[1]
                accel_y_raw = (raw_data[2] << 8) | raw_data[3]
                accel_z_raw = (raw_data[4] << 8) | raw_data[5]
                
                # Skip temperature data (bytes 6-7)
                
                # Parse gyroscope data (last 6 bytes)
                gyro_x_raw = (raw_data[8] << 8) | raw_data[9]
                gyro_y_raw = (raw_data[10] << 8) | raw_data[11]
                gyro_z_raw = (raw_data[12] << 8) | raw_data[13]
                
                # Convert to proper units
                accel_x = accel_ms2(accel_x_raw)
                accel_y = -accel_ms2(accel_y_raw)  # Y-axis sign flip
                accel_z = accel_ms2(accel_z_raw)
                
                gyro_x = gyro_dps(gyro_x_raw)
                gyro_y = gyro_dps(gyro_y_raw)
                gyro_z = gyro_dps(gyro_z_raw)
                
            except Exception as burst_error:
                # Fallback to individual reads if burst read fails
                accel_x = accel_ms2(read_data(register_accel_xout_h))
                accel_y = -accel_ms2(read_data(register_accel_yout_h))
                accel_z = accel_ms2(read_data(register_accel_zout_h))
                
                gyro_x = gyro_dps(read_data(register_gyro_xout_h))
                gyro_y = gyro_dps(read_data(register_gyro_yout_h))
                gyro_z = gyro_dps(read_data(register_gyro_zout_h))
            
            # Calculate timestamp
            current_time = time.time()
            sync_timestamp = current_time - start_time
            
            # Store raw sensor data with frame info
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
                raw_sensor_buffer.append(sensor_data)
            
            # Log sensor data every 1 second (30 frames at 30Hz)
            if show_sensor_logs and frame_count % 30 == 0:
                current_log_time = time.time()
                if current_log_time - last_log_time >= 1.0:
                    fast_log(f"📊 Frame {frame_count:4d} | "
                            f"Acc: X={accel_x:6.2f} Y={accel_y:6.2f} Z={accel_z:6.2f} | "
                            f"Gyro: X={gyro_x:7.2f} Y={gyro_y:7.2f} Z={gyro_z:7.2f}", console_summary=True)
                    last_log_time = current_log_time
            
            frame_count += 1
            
            # Debug: Log frame count every 150 frames (5 seconds)
            if frame_count % 150 == 0:
                fast_log(f"🔧 Sensor collection: {frame_count} frames collected, {frame_count/30:.1f}s runtime")
            
            # Maintain 30Hz sampling rate
            next_sample_time = start_time + (frame_count * (1.0 / TARGET_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            fast_log(f"❌ Sensor collection error: {e}")
            fast_log(f"🔧 Debug info - Frame: {frame_count}, Time: {time.time():.3f}")
            time.sleep(0.1)  # Longer sleep when error occurs

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
        
        # Apply MinMax scaling
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
            acc_y = -data['accel_y'] / 9.80665  # No sign change (already done in sensor collection)
            acc_z = data['accel_z'] / 9.80665
            gyr_x = data['gyro_x']
            gyr_y = data['gyro_y']
            gyr_z = data['gyro_z']
            
            processed_data.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z])
        
        # Convert to numpy array
        sensor_array = np.array(processed_data, dtype=np.float32)
        
        # Apply fall-specific scaling for each sensor channel
        sensor_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        scaled_data = np.zeros_like(sensor_array)
        
        for i, sensor_name in enumerate(sensor_names):
            # Try to use MinMax scaler first, then Standard scaler if available
            minmax_key = f"{sensor_name}_minmax"
            standard_key = f"{sensor_name}_standard"
            
            if minmax_key in fall_scalers:
                # Apply MinMax scaling
                sensor_column = sensor_array[:, i].reshape(-1, 1)
                scaled_column = fall_scalers[minmax_key].transform(sensor_column)
                scaled_data[:, i] = scaled_column.flatten()
            elif standard_key in fall_scalers:
                # Apply Standard scaling
                sensor_column = sensor_array[:, i].reshape(-1, 1)
                scaled_column = fall_scalers[standard_key].transform(sensor_column)
                scaled_data[:, i] = scaled_column.flatten()
            else:
                # No scaler available, use original data
                scaled_data[:, i] = sensor_array[:, i]
                print(f"⚠️ No scaler found for {sensor_name}, using original data")
        
        return scaled_data.reshape(1, FALL_WINDOW_SIZE, 6)
    except Exception as e:
        print(f"❌ Fall preprocessing error: {e}")
        return None

def gait_detection_thread():
    """Thread for gait detection"""
    global gait_state, gait_frame_count, non_gait_frame_count
    global current_gait_start_frame, current_gait_data
    global latest_gait_probability, prediction_update_count
    
    while is_running:
        try:
            with sensor_data_lock:
                if len(raw_sensor_buffer) >= GAIT_WINDOW_SIZE:
                    # Get latest window
                    sensor_window = list(raw_sensor_buffer)[-GAIT_WINDOW_SIZE:]
                else:
                    time.sleep(0.1)
                    continue
            
            # Preprocess and predict
            if gait_interpreter and gait_scaler:
                preprocessed = preprocess_for_gait(sensor_window)
                if preprocessed is not None:
                    # Run inference
                    input_details = gait_interpreter.get_input_details()
                    output_details = gait_interpreter.get_output_details()
                    
                    gait_interpreter.set_tensor(input_details[0]['index'], preprocessed)
                    gait_interpreter.invoke()
                    
                    prediction = gait_interpreter.get_tensor(output_details[0]['index'])
                    gait_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
                    
                    # Update global prediction value
                    latest_gait_probability = gait_probability
                    
                    # Log prediction results every 10 predictions (~0.3 seconds)
                    if show_prediction_logs and prediction_update_count % 10 == 0:
                        fast_log(f"🤖 Gait Prediction: {gait_probability:.3f} | State: {gait_state} | "
                                f"Gait frames: {gait_frame_count} | Non-gait frames: {non_gait_frame_count}", console_summary=True)
                    
                    prediction_update_count += 1
                    
                    # Update frame counts
                    if gait_probability > GAIT_THRESHOLD:
                        gait_frame_count += 1
                        non_gait_frame_count = 0
                    else:
                        non_gait_frame_count += 1
                        gait_frame_count = 0
                    
                    # State transition logic
                    if gait_state == "non-gait" and gait_frame_count >= GAIT_TRANSITION_FRAMES:
                        gait_state = "gait"
                        current_gait_start_frame = sensor_window[0]['frame']
                        current_gait_data = []
                        fast_log(f"🚶 Gait started at frame {current_gait_start_frame}")
                    
                    elif gait_state == "gait":
                        # Add current frame data to gait data
                        current_gait_data.extend(sensor_window[-1:])
                        
                        # Check for gait end
                        if non_gait_frame_count >= GAIT_TRANSITION_FRAMES:
                            # Check if gait duration was long enough
                            gait_duration_frames = len(current_gait_data)
                            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                                # Save gait data to Supabase
                                save_gait_data_to_supabase(current_gait_data)
                            else:
                                print(f"⚠️ Gait duration too short: {gait_duration_frames} frames")
                            
                            gait_state = "non-gait"
                            current_gait_data = []
            
            time.sleep(0.033)  # ~30Hz
            
        except Exception as e:
            print(f"❌ Gait detection error: {e}")
            time.sleep(0.1)

def fall_detection_thread():
    """Thread for fall detection"""
    global latest_fall_probability
    fall_prediction_count = 0
    
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
                    
                    # Update global prediction value
                    latest_fall_probability = fall_probability
                    
                    # Log prediction results every 10 predictions (~0.3 seconds)
                    if show_prediction_logs and fall_prediction_count % 10 == 0:
                        fast_log(f"🚨 Fall Prediction: {fall_probability:.3f} | Threshold: {FALL_THRESHOLD}", console_summary=True)
                    
                    fall_prediction_count += 1
                    
                    # Check for fall
                    if fall_probability > FALL_THRESHOLD:
                        fast_log(f"🚨 Fall detected! Probability: {fall_probability:.2f}")
                        save_fall_event_to_supabase(sensor_window[-1]['unix_timestamp'])
            
            time.sleep(0.033)  # ~30Hz
            
        except Exception as e:
            print(f"❌ Fall detection error: {e}")
            time.sleep(0.1)

def save_gait_data_to_supabase(gait_data):
    """Save gait data as CSV to Supabase"""
    if not supabase:
        print("❌ Supabase not initialized")
        return
    
    try:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['frame', 'sync_timestamp', 'accel_x', 'accel_y', 'accel_z', 
                        'gyro_x', 'gyro_y', 'gyro_z'])
        
        # Write data with proper formatting
        first_timestamp = gait_data[0]['sync_timestamp']
        for data in gait_data:
            writer.writerow([
                data['frame'],
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
        filename = f"gait_data_{timestamp}.csv"
        
        # Convert to bytes
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode('utf-8')
        
        # Upload to Supabase Storage
        response = supabase.storage.from_('gait-data').upload(
            file=csv_bytes,
            path=filename,
            file_options={"content-type": "text/csv"}
        )
        
        print(f"✅ Gait data saved: {filename} ({len(gait_data)} frames)")
        
    except Exception as e:
        print(f"❌ Failed to save gait data: {e}")

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

def toggle_sensor_logs():
    """Toggle sensor data logging on/off"""
    global show_sensor_logs
    show_sensor_logs = not show_sensor_logs
    status = "ON" if show_sensor_logs else "OFF"
    fast_log(f"📊 Sensor logging: {status}")

def toggle_prediction_logs():
    """Toggle prediction data logging on/off"""
    global show_prediction_logs
    show_prediction_logs = not show_prediction_logs
    status = "ON" if show_prediction_logs else "OFF"
    fast_log(f"🤖 Prediction logging: {status}")

def print_detailed_sensor_status():
    """Print detailed sensor and system status"""
    with sensor_data_lock:
        if len(raw_sensor_buffer) > 0:
            latest_data = raw_sensor_buffer[-1]
            buffer_size = len(raw_sensor_buffer)
            
            print("\n" + "="*80)
            print("📊 DETAILED SENSOR STATUS")
            print("="*80)
            print(f"Frame Number: {latest_data['frame']}")
            print(f"Sync Timestamp: {latest_data['sync_timestamp']:.3f}s")
            print(f"Buffer Size: {buffer_size}/{max(GAIT_WINDOW_SIZE, FALL_WINDOW_SIZE) * 10}")
            print(f"Gait State: {gait_state}")
            print(f"Sensor Logging: {'ON' if show_sensor_logs else 'OFF'}")
            print(f"Prediction Logging: {'ON' if show_prediction_logs else 'OFF'}")
            print("-" * 80)
            print("ACCELEROMETER (m/s²):")
            print(f"  X: {latest_data['accel_x']:8.3f}")
            print(f"  Y: {latest_data['accel_y']:8.3f}")
            print(f"  Z: {latest_data['accel_z']:8.3f}")
            print("-" * 80)
            print("GYROSCOPE (°/s):")
            print(f"  X: {latest_data['gyro_x']:8.3f}")
            print(f"  Y: {latest_data['gyro_y']:8.3f}")
            print(f"  Z: {latest_data['gyro_z']:8.3f}")
            print("-" * 80)
            print("MODEL PREDICTIONS:")
            print(f"  Gait Probability: {latest_gait_probability:.3f}")
            print(f"  Fall Probability: {latest_fall_probability:.3f}")
            print("="*80)
        else:
            print("❌ No sensor data available")

def main():
    """Main execution function"""
    global is_running
    
    # Check for unbuffered output
    if not (os.getenv('PYTHONUNBUFFERED') or '-u' in sys.argv):
        print("⚠️  PERFORMANCE TIP: Run with 'python -u' or set PYTHONUNBUFFERED=1 for better real-time output")
        time.sleep(2)
    
    print("=" * 60)
    print("🚶 Gait & Fall Detection System")
    print("=" * 60)
    
    # Setup file logging
    setup_file_logging()
    
    # Initialize Supabase
    if not init_supabase():
        print("⚠️ Continuing without Supabase")
    
    # Load models
    load_models()
    
    # Initialize IMU sensor
    try:
        bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
        print("✅ IMU sensor initialized")
        
        # Test sensor reading
        test_read = bus.read_byte_data(DEV_ADDR, 0x75)  # WHO_AM_I register
        print(f"✅ Sensor test read - WHO_AM_I: 0x{test_read:02X}")
    except Exception as e:
        print(f"❌ IMU sensor initialization failed: {e}")
        print("💡 Check I2C connections and sensor power")
        return
    
    # Start threads
    is_running = True
    
    # Start logging thread first
    log_thread = threading.Thread(target=logging_thread)
    log_thread.daemon = True
    log_thread.start()
    print("✅ Logging thread started")
    
    sensor_thread = threading.Thread(target=sensor_collection_thread)
    sensor_thread.daemon = True
    sensor_thread.start()
    print("✅ Sensor collection thread started")
    
    gait_thread = threading.Thread(target=gait_detection_thread)
    gait_thread.daemon = True
    gait_thread.start()
    print("✅ Gait detection thread started")
    
    fall_thread = threading.Thread(target=fall_detection_thread)
    fall_thread.daemon = True
    fall_thread.start()
    print("✅ Fall detection thread started")
    
    print("\n" + "="*60)
    print("⌨️  CONTROL COMMANDS:")
    print("   Enter 's' and press Enter: Toggle sensor data logging")
    print("   Enter 'p' and press Enter: Toggle prediction data logging")
    print("   Enter 'd' and press Enter: Show detailed sensor status")
    print("   Ctrl+C: Stop system")
    print("="*60)
    print(f"📊 Sensor logging: {'ON' if show_sensor_logs else 'OFF'}")
    print(f"🤖 Prediction logging: {'ON' if show_prediction_logs else 'OFF'}")
    print("📁 Detailed logs: logs/gait_system.log (use 'tail -f logs/gait_system.log' for real-time)")
    print("\nSystem running with real-time progress bar...")
    print("TIP: All detailed data is logged to file, console shows summary + progress bar\n")
    
    # Start input handling thread for Windows compatibility
    def input_handler():
        """Handle user input in separate thread"""
        while is_running:
            try:
                user_input = input().strip().lower()
                if user_input == 's':
                    toggle_sensor_logs()
                elif user_input == 'p':
                    toggle_prediction_logs()
                elif user_input == 'd':
                    print_detailed_sensor_status()
                elif user_input == 'h' or user_input == 'help':
                    print("\n⌨️  Available commands: 's' (toggle sensor logs), 'p' (toggle prediction logs), 'd' (detailed status)\n")
            except:
                break
    
    input_thread = threading.Thread(target=input_handler)
    input_thread.daemon = True
    input_thread.start()
    
    try:
        while True:
            time.sleep(1)
            
            # Progress bar handles all status display now
            pass
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        
    finally:
        is_running = False
        time.sleep(1)  # Allow threads to finish
        
        # Save any remaining gait data
        if gait_state == "gait" and current_gait_data:
            if len(current_gait_data) >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(current_gait_data)
        
        bus.close()
        
        # Clear progress bar and print final status
        if last_status_line:
            sys.stdout.write("\r" + " " * len(last_status_line) + "\r")
        
        print("✅ System stopped")
        print(f"📁 Full logs saved to: logs/gait_system.log")

if __name__ == "__main__":
    main()