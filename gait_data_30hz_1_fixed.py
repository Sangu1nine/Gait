"""
보행 및 낙상 감지 IMU 센서 데이터 수집 프로그램 (개선된 버퍼 처리)
MODIFIED 2025-01-30: 버퍼 처리 로직 개선 - 안정적인 프레임 처리
Features:
- 30Hz IMU 센서 데이터 수집 (멀티스레드)
- 실시간 보행 감지 (TensorFlow Lite) - 개선된 버퍼 관리
- 실시간 낙상 감지 (TensorFlow Lite)
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
TARGET_HZ = 30   # Sampling rate
GAIT_THRESHOLD = 0.5  # Gait detection threshold
FALL_THRESHOLD = 0.5  # Fall detection threshold

# State transition parameters
GAIT_TRANSITION_FRAMES = 60  # 2 seconds at 30Hz
MIN_GAIT_DURATION_FRAMES = 300  # 10 seconds at 30Hz

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = None

# Global variables for sensor data collection
sensor_data_lock = threading.Lock()
raw_sensor_buffer = deque(maxlen=max(GAIT_WINDOW_SIZE, FALL_WINDOW_SIZE) * 5)  # 더 작은 버퍼 크기
is_running = False

# Gait detection variables - 개선된 구조
gait_interpreter = None
gait_scaler = None
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

def sensor_collection_thread():
    """Thread for collecting sensor data at 30Hz"""
    global raw_sensor_buffer, is_running
    
    start_time = time.time()
    frame_count = 0
    
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
            
            frame_count += 1
            
            # Maintain 30Hz sampling rate
            next_sample_time = start_time + (frame_count * (1.0 / TARGET_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"❌ Sensor collection error: {e}")
            time.sleep(0.01)

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
            acc_y = -data['accel_y'] / 9.80665  # Sign change
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
    """Thread for gait detection - 개선된 간단한 방식"""
    global gait_state, gait_consecutive_count, non_gait_consecutive_count
    global current_gait_data, current_gait_start_time, last_prediction_frame
    
    print("🚶 Gait detection thread initialized")
    
    while is_running:
        try:
            # Get available sensor data
            with sensor_data_lock:
                if len(raw_sensor_buffer) < GAIT_WINDOW_SIZE:
                    time.sleep(0.01)
                    continue
                
                # Simple sliding window - just get the latest window
                sensor_window = list(raw_sensor_buffer)[-GAIT_WINDOW_SIZE:]
            
            # Get current frame number
            current_frame = sensor_window[-1]['frame']
            
            # Skip if already processed this frame
            if current_frame <= last_prediction_frame:
                time.sleep(0.01)
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
                    
                    # Update consecutive counts
                    if gait_probability > GAIT_THRESHOLD:
                        gait_consecutive_count += 1
                        non_gait_consecutive_count = 0
                    else:
                        non_gait_consecutive_count += 1
                        gait_consecutive_count = 0
                    
                    # State transition logic
                    update_gait_state_simple(sensor_window[-1], gait_probability)
                    
                    last_prediction_frame = current_frame
            
            time.sleep(0.033)  # ~30Hz
            
        except Exception as e:
            print(f"❌ Gait detection error: {e}")
            time.sleep(0.1)

def update_gait_state_simple(latest_sensor_data, gait_probability):
    """간단하고 효율적인 보행 상태 업데이트"""
    global gait_state, current_gait_data, current_gait_start_time
    
    if gait_state == "non-gait":
        # Gait 시작 조건: 연속으로 GAIT_TRANSITION_FRAMES 만큼 gait로 예측됨
        if gait_consecutive_count >= GAIT_TRANSITION_FRAMES:
            gait_state = "gait"
            current_gait_start_time = latest_sensor_data['unix_timestamp']
            current_gait_data = deque()  # 새로운 gait 데이터 시작
            print(f"🚶 Gait started at frame {latest_sensor_data['frame']} (confidence: {gait_consecutive_count}/{GAIT_TRANSITION_FRAMES})")
    
    elif gait_state == "gait":
        # 보행 중인 경우 데이터 수집
        current_gait_data.append(latest_sensor_data)
        
        # Gait 종료 조건: 연속으로 GAIT_TRANSITION_FRAMES 만큼 non-gait로 예측됨
        if non_gait_consecutive_count >= GAIT_TRANSITION_FRAMES:
            # 보행 데이터 저장 체크
            gait_duration_frames = len(current_gait_data)
            gait_duration_seconds = gait_duration_frames / TARGET_HZ
            
            print(f"🛑 Gait ended at frame {latest_sensor_data['frame']} (duration: {gait_duration_frames} frames, {gait_duration_seconds:.1f}s)")
            
            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(list(current_gait_data))
                print(f"✅ Gait data saved ({gait_duration_frames} frames)")
            else:
                print(f"⚠️ Gait duration too short: {gait_duration_frames} frames ({gait_duration_seconds:.1f}s < {MIN_GAIT_DURATION_FRAMES/TARGET_HZ:.1f}s)")
            
            # 상태 리셋
            gait_state = "non-gait"
            current_gait_data = deque()
            current_gait_start_time = None

def fall_detection_thread():
    """Thread for fall detection"""
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

def main():
    """Main execution function"""
    global is_running
    
    print("=" * 60)
    print("🚶 Gait & Fall Detection System (Fixed Buffer)")
    print("=" * 60)
    
    # Initialize Supabase
    if not init_supabase():
        print("⚠️ Continuing without Supabase")
    
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
    print("✅ Sensor collection thread started")
    
    gait_thread = threading.Thread(target=gait_detection_thread)
    gait_thread.daemon = True
    gait_thread.start()
    print("✅ Gait detection thread started")
    
    fall_thread = threading.Thread(target=fall_detection_thread)
    fall_thread.daemon = True
    fall_thread.start()
    print("✅ Fall detection thread started")
    
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Print status every 5 seconds
            if int(time.time()) % 5 == 0:
                with sensor_data_lock:
                    buffer_size = len(raw_sensor_buffer)
                
                gait_data_size = len(current_gait_data) if current_gait_data else 0
                
                print(f"📊 Status - Buffer: {buffer_size}, State: {gait_state}, "
                      f"Gait count: +{gait_consecutive_count}/-{non_gait_consecutive_count}, "
                      f"Gait frames: {gait_data_size}, Last frame: {last_prediction_frame}")
                
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