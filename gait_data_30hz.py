"""
GAIT Detection IMU Sensor Data Collection Program (30Hz)
Created: 2025-01-29
Modified: 2025-01-29 - Added gait detection functionality
MODIFIED 2025-01-29: Changed all log messages from Korean to English - for international compatibility
MODIFIED 2025-01-29: Added Walking-Bout Segmentation Logic - implements state-based bout detection

Features:
- IMU sensor data collection at 30Hz
- Real-time gait detection using Stage1 TensorFlow Lite model
- Walking-Bout Segmentation Logic (IDLE → WALKING → POST-WALK)
- WiFi data transmission only during WALKING state
- Save data to separate CSV files for each walking bout
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

# WiFi communication settings
WIFI_SERVER_IP = '172.20.10.12'  # Local PC IP address (modified)
WIFI_SERVER_PORT = 5000  # Communication port
wifi_client = None
wifi_connected = False
send_data_queue = []

# Gait detection related settings
MODEL_PATH = "models/gait_detection/model.tflite"
SCALER_PATH = "scalers/gait_detection"  # Directory containing scaler file
WINDOW_SIZE = 60  # Window size for Stage1 model
TARGET_HZ = 30   # Sampling rate
GAIT_THRESHOLD = 0.5  # Gait detection threshold (default value)

# Walking-Bout Segmentation parameters
N1 = 60  # Minimum consecutive gait frames to start walking (≈ 2s @30Hz)
N2 = 60  # Minimum consecutive non-gait frames to end walking (≈ 2s @30Hz)
# Note: Using GAIT_THRESHOLD (0.2) for gait probability threshold

# Walking-Bout Segmentation states
IDLE = "IDLE"
WALKING = "WALKING"
POST_WALK = "POST_WALK"

# Gait detection global variables
interpreter = None
minmax_scaler = None
sensor_buffer = deque(maxlen=WINDOW_SIZE)
gait_detection_enabled = False
last_gait_status = "non_gait"
gait_count = 0
non_gait_count = 0

# Walking-Bout Segmentation global variables
current_state = IDLE
consecutive_gait_frames = 0
consecutive_non_gait_frames = 0
current_bout_data = []
bout_counter = 0
current_csv_filename = None
bout_start_time = None

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

def load_gait_detection_model():
    """Load gait detection model and scaler"""
    global interpreter, minmax_scaler, gait_detection_enabled
    
    try:
        # Load TensorFlow Lite model
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            print(f"✅ TFLite model loaded successfully: {MODEL_PATH}")
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
        
        # Load MinMax scaler
        scaler_file = os.path.join(SCALER_PATH, "minmax_scaler.pkl")
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                minmax_scaler = pickle.load(f)
            print(f"✅ MinMax scaler loaded successfully: {scaler_file}")
        else:
            print(f"❌ Scaler file not found: {scaler_file}")
            return False
        
        gait_detection_enabled = True
        print("🚶 Gait detection system activated")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        return False

def predict_gait(sensor_data):
    """Predict gait detection from sensor data"""
    global interpreter, minmax_scaler, last_gait_status
    
    if not gait_detection_enabled or len(sensor_data) != WINDOW_SIZE:
        return "unknown", 0.0
    
    try:
        # Data preprocessing (refer to stage1_preprocessing.py)
        sensor_array = np.array(sensor_data, dtype=np.float32).reshape(1, WINDOW_SIZE, 6)
        
        # Apply MinMax scaling
        n_samples, n_frames, n_features = sensor_array.shape
        sensor_reshaped = sensor_array.reshape(-1, n_features)
        sensor_scaled = minmax_scaler.transform(sensor_reshaped)
        sensor_scaled = sensor_scaled.reshape(n_samples, n_frames, n_features).astype(np.float32)
        
        # TensorFlow Lite model inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], sensor_scaled)
        interpreter.invoke()
        
        # Get prediction results
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Convert probability to binary classification (using threshold)
        gait_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
        predicted_class = "gait" if gait_probability > GAIT_THRESHOLD else "non_gait"
        
        last_gait_status = predicted_class
        return predicted_class, float(gait_probability)
        
    except Exception as e:
        print(f"⚠️  Gait detection error: {str(e)}")
        return "unknown", 0.0

def connect_wifi():
    """Setup WiFi connection"""
    global wifi_client, wifi_connected
    try:
        wifi_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        wifi_client.settimeout(5.0)  # 5 second timeout
        wifi_client.connect((WIFI_SERVER_IP, WIFI_SERVER_PORT))
        wifi_connected = True
        print(f"✅ WiFi connection successful: {WIFI_SERVER_IP}:{WIFI_SERVER_PORT}")
        return True
    except socket.timeout:
        print(f"❌ WiFi connection timeout: {WIFI_SERVER_IP}:{WIFI_SERVER_PORT}")
        wifi_connected = False
        return False
    except Exception as e:
        print(f"❌ WiFi connection failed: {str(e)}")
        wifi_connected = False
        return False

def send_data_thread():
    """Data transmission thread"""
    global send_data_queue, wifi_client, wifi_connected
    
    while wifi_connected:
        if len(send_data_queue) > 0:
            try:
                # Get data from queue
                sensor_data = send_data_queue.pop(0)
                # Convert to JSON format and send
                data_json = json.dumps(sensor_data)
                wifi_client.sendall((data_json + '\n').encode('utf-8'))
            except Exception as e:
                print(f"❌ Data transmission error: {str(e)}")
                wifi_connected = False
                break
        else:
            time.sleep(0.001)

def close_wifi():
    """Close WiFi connection"""
    global wifi_client, wifi_connected
    if wifi_client:
        try:
            wifi_client.close()
            print("✅ WiFi connection closed")
        except:
            pass
    wifi_connected = False

def start_new_bout():
    """Start a new walking bout"""
    global current_bout_data, bout_counter, current_csv_filename, bout_start_time
    
    bout_counter += 1
    current_bout_data = []
    bout_start_time = time.time()
    
    # Generate filename for current bout
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_csv_filename = f"walking_bout_{bout_counter:03d}_{timestamp}.csv"
    
    print(f"🚶 Walking bout {bout_counter} started - File: {current_csv_filename}")

def save_current_bout():
    """Save current walking bout to CSV file"""
    global current_bout_data, current_csv_filename, bout_start_time
    
    if len(current_bout_data) > 0 and current_csv_filename:
        # Create dataframe and save
        columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'Timestamp', 'GaitStatus']
        df = pd.DataFrame(current_bout_data, columns=columns)
        df.to_csv(current_csv_filename, index=False)
        
        # Calculate bout statistics
        bout_duration = time.time() - bout_start_time if bout_start_time else 0
        gait_frames = sum(1 for row in current_bout_data if row[7] == "gait")
        total_frames = len(current_bout_data)
        gait_percentage = (gait_frames / total_frames * 100) if total_frames > 0 else 0
        
        print(f"💾 Walking bout {bout_counter} saved:")
        print(f"   📄 File: {current_csv_filename}")
        print(f"   ⏱️  Duration: {bout_duration:.2f}s")
        print(f"   📊 Total frames: {total_frames}")
        print(f"   🚶 Gait frames: {gait_frames} ({gait_percentage:.1f}%)")
        
        # Reset bout data
        current_bout_data = []
        current_csv_filename = None
        bout_start_time = None

def update_walking_bout_state(gait_status, gait_probability):
    """Update walking-bout segmentation state"""
    global current_state, consecutive_gait_frames, consecutive_non_gait_frames
    
    # Update consecutive frame counters
    if gait_status == "gait":
        consecutive_gait_frames += 1
        consecutive_non_gait_frames = 0
    elif gait_status == "non_gait":
        consecutive_non_gait_frames += 1
        consecutive_gait_frames = 0
    else:
        # Unknown status - don't change counters
        return current_state, False
    
    previous_state = current_state
    should_transmit = False
    
    # State transition logic
    if current_state == IDLE:
        # IDLE → WALKING: N1 consecutive gait frames
        if consecutive_gait_frames >= N1:
            current_state = WALKING
            start_new_bout()
            should_transmit = True
            print(f"🔄 State change: {previous_state} → {current_state}")
    
    elif current_state == WALKING:
        # Continue transmission if gait probability > threshold
        if gait_probability > GAIT_THRESHOLD:
            should_transmit = True
        
        # WALKING → POST_WALK: N2 consecutive non-gait frames
        if consecutive_non_gait_frames >= N2:
            current_state = POST_WALK
            print(f"🔄 State change: {previous_state} → {current_state}")
    
    elif current_state == POST_WALK:
        # POST_WALK → IDLE: save current bout and return to idle
        save_current_bout()
        current_state = IDLE
        consecutive_gait_frames = 0
        consecutive_non_gait_frames = 0
        print(f"🔄 State change: {previous_state} → {current_state}")
    
    return current_state, should_transmit

def main():
    """Main execution function"""
    global sensor_buffer, gait_count, non_gait_count, current_bout_data
    
    print("=" * 60)
    print("🚶 GAIT Detection IMU Sensor Data Collection Program (30Hz)")
    print("📊 Walking-Bout Segmentation Logic Enabled")
    print("=" * 60)
    
    # Load gait detection model
    if not load_gait_detection_model():
        print("⚠️  Unable to load gait detection model. All data will be transmitted.")
    
    # Initialize sensor
    bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
    
    # Set base filename for logging (all sensor data)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gait_log_all_data_{timestamp}.csv"
    all_data = []  # Store all data for logging
    
    print(f"📄 Complete data log file: {log_filename}")
    print(f"🎯 Gait detection threshold: {GAIT_THRESHOLD}")
    print(f"📊 Walking-Bout parameters: N1={N1}, N2={N2}, θ={GAIT_THRESHOLD}")
    print(f"🔄 Initial state: {current_state}")
    print("Press Ctrl+C to stop collection")
    
    # Attempt WiFi connection
    wifi_thread = None
    if connect_wifi():
        wifi_thread = threading.Thread(target=send_data_thread)
        wifi_thread.daemon = True
        wifi_thread.start()
    
    # Initial time
    start_time = time.time()
    sample_count = 0
    
    try:
        while True:
            # Calculate current sample time
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Read IMU sensor data
            accel_x = accel_ms2(read_data(register_accel_xout_h))
            accel_y = accel_ms2(read_data(register_accel_yout_h))
            accel_z = accel_ms2(read_data(register_accel_zout_h))
            
            gyro_x = gyro_dps(read_data(register_gyro_xout_h))
            gyro_y = gyro_dps(read_data(register_gyro_yout_h))
            gyro_z = gyro_dps(read_data(register_gyro_zout_h))
            
            # Sensor data (same order as Stage1 preprocessing)
            sensor_row = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            
            # Add to buffer for gait detection
            sensor_buffer.append(sensor_row)
            
            # Gait detection (when buffer is sufficiently filled)
            gait_status = "unknown"
            gait_probability = 0.0
            if len(sensor_buffer) == WINDOW_SIZE:
                gait_status, gait_probability = predict_gait(list(sensor_buffer))
                
                # Update statistics
                if gait_status == "gait":
                    gait_count += 1
                elif gait_status == "non_gait":
                    non_gait_count += 1
                
                # Update walking-bout state and determine transmission
                state, should_transmit = update_walking_bout_state(gait_status, gait_probability)
                
                # WiFi transmission based on walking-bout state
                if should_transmit and wifi_connected:
                    sensor_data_wifi = {
                        'timestamp': elapsed,
                        'accel': {'x': accel_x, 'y': -accel_y, 'z': accel_z},  # Y-axis inverted
                        'gyro': {'x': gyro_x, 'y': -gyro_y, 'z': gyro_z},      # Y-axis inverted
                        'gait_status': gait_status,
                        'bout_state': state,
                        'bout_number': bout_counter
                    }
                    send_data_queue.append(sensor_data_wifi)
                
                # Add to current bout data if in WALKING state
                if current_state == WALKING:
                    current_bout_data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, elapsed, gait_status])
            
            # Save all data to complete log (including gait status and state)
            all_data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, elapsed, gait_status, current_state])
            sample_count += 1
            
            # Output progress every 30 samples
            if sample_count % 30 == 0:
                total_predictions = gait_count + non_gait_count
                gait_percentage = (gait_count / total_predictions * 100) if total_predictions > 0 else 0
                
                print(f"📊 Samples: {sample_count}, Time: {elapsed:.2f}s, Sampling rate: {sample_count/elapsed:.2f}Hz")
                print(f"🏃 Acceleration(m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                print(f"🔄 Gyroscope(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                print(f"🚶 Gait status: {gait_status} (Gait ratio: {gait_percentage:.1f}%)")
                print(f"🔄 Walking-Bout State: {current_state}")
                print(f"   📊 Consecutive gait: {consecutive_gait_frames}, non-gait: {consecutive_non_gait_frames}")
                print(f"   📄 Current bout: {bout_counter}, bout frames: {len(current_bout_data)}")
                if wifi_connected:
                    transmitted = "Transmitted" if current_state == WALKING and len(sensor_buffer) == WINDOW_SIZE else "Not transmitted"
                    print(f"📡 WiFi: Connected, Queue length: {len(send_data_queue)}, Status: {transmitted}")
                else:
                    print("📡 WiFi: Not connected")
                print("-" * 50)
            
            # Maintain sampling rate (30Hz)
            next_sample_time = start_time + (sample_count * (1.0 / TARGET_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        
    finally:
        # Save any remaining bout data
        if current_state == WALKING and len(current_bout_data) > 0:
            save_current_bout()
        
        # Create and save complete log dataframe
        columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'Timestamp', 'GaitStatus', 'BoutState']
        df_all = pd.DataFrame(all_data, columns=columns)
        df_all.to_csv(log_filename, index=False)
        
        # Output final statistics
        total_samples = len(df_all)
        total_predictions = gait_count + non_gait_count
        gait_percentage = (gait_count / total_predictions * 100) if total_predictions > 0 else 0
        
        print("=" * 60)
        print("📊 Collection Complete Statistics")
        print("=" * 60)
        print(f"📄 Complete log file: {log_filename}")
        print(f"📈 Total samples: {total_samples}")
        print(f"🚶 Total walking bouts detected: {bout_counter}")
        print(f"🔄 Final state: {current_state}")
        print(f"🚶 Gait detected: {gait_count} times ({gait_percentage:.1f}%)")
        print(f"🏃 Non-gait detected: {non_gait_count} times ({100-gait_percentage:.1f}%)")
        print(f"⏱️  Total collection time: {elapsed:.2f}s")
        print(f"📊 Average sampling rate: {total_samples/elapsed:.2f}Hz")
        print("=" * 60)
        
        # Resource cleanup
        close_wifi()
        bus.close()
        print("✅ All resources cleaned up")

if __name__ == "__main__":
    main() 