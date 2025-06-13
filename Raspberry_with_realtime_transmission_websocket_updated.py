"""
Enhanced Raspberry Pi Fall Detection System
- Simplified code structure
- Stable data transmission
- Fall data guarantee system
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
    print("SMBus2 library not found. Please run 'pip install smbus2'")
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
MODEL_PATH = 'models/fall_detection/fall_detection.tflite'
SCALERS_DIR = 'scalers/fall_detection'
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
    """Safe data transmission manager"""
    def __init__(self):
        self.imu_queue = queue.Queue(maxsize=100)  # Size reduced for 10Hz
        self.fall_queue = queue.Queue(maxsize=100)  # Separate queue for fall data
        self.websocket = None
        self.connected = False
        
    def add_imu_data(self, data):
        """Add IMU data (remove old data if queue is full)"""
        try:
            self.imu_queue.put_nowait(data)
        except queue.Full:
            try:
                self.imu_queue.get_nowait()  # Remove old data
                self.imu_queue.put_nowait(data)
            except queue.Empty:
                pass
    
    def add_fall_data(self, data):
        """Add fall data (prevent absolute loss)"""
        try:
            self.fall_queue.put_nowait(data)
            print(f"🚨 Fall data added to queue! Waiting: {self.fall_queue.qsize()} items")
        except queue.Full:
            print("❌ Fall data queue full! Emergency processing needed")
    
    async def send_loop(self):
        """Data transmission loop"""
        while True:
            try:
                # Priority processing for fall data
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data, is_fall=True)
                
                # IMU data processing (only when connected)
                elif self.connected and not self.imu_queue.empty():
                    imu_data = self.imu_queue.get_nowait()
                    await self._send_data(imu_data, is_fall=False)
                
                await asyncio.sleep(0.01)  # Adjust transmission interval for 10Hz
                
            except Exception as e:
                print(f"Transmission loop error: {e}")
                await asyncio.sleep(1)
    
    async def _send_data(self, data, is_fall=False):
        """Actual data transmission"""
        if not self.websocket:
            if is_fall:
                # Add fall data back to queue
                self.fall_queue.put_nowait(data)
            return
        
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            await self.websocket.send(json_data)
            
            if is_fall:
                print(f"🚨 Fall data transmission successful! Confidence: {data['data'].get('confidence_score', 0):.2%}")
                
        except Exception as e:
            print(f"Data transmission failed: {e}")
            if is_fall:
                # Add fall data back to queue
                self.fall_queue.put_nowait(data)

class SimpleSensor:
    """Simplified sensor class"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("Sensor library not available")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        
        # Load scalers
        self.scalers = self._load_scalers()
        print("Sensor initialization complete")
    
    def _load_scalers(self):
        """Load scalers"""
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
                print(f"Failed to load scaler {feature}: {e}")
        
        return scalers
    
    def _read_word_2c(self, reg):
        """Read two's complement value"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val
    
    def get_data(self):
        """Read and normalize sensor data"""
        # Read raw data
        raw_data = []
        for reg in ACCEL_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_ACCEL)
        for reg in GYRO_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_GYRO)
        
        # Normalize
        if self.scalers:
            features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            normalized = []
            for i, feature in enumerate(features):
                val = raw_data[i]
                
                # Standard scaling
                if f"{feature}_standard" in self.scalers:
                    scaler = self.scalers[f"{feature}_standard"]
                    val = (val - scaler.mean_[0]) / scaler.scale_[0]
                
                # MinMax scaling
                if f"{feature}_minmax" in self.scalers:
                    scaler = self.scalers[f"{feature}_minmax"]
                    val = val * scaler.scale_[0] + scaler.min_[0]
                
                normalized.append(val)
            return np.array(normalized)
        
        return np.array(raw_data)

class SimpleFallDetector:
    """Simplified fall detector"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Fall detection model loaded successfully")
    
    def add_data(self, data):
        """Add data"""
        self.buffer.append(data)
        self.counter += 1
    
    def should_predict(self):
        """Check prediction timing"""
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0
    
    def predict(self):
        """Fall prediction"""
        if len(self.buffer) < SEQ_LENGTH:
            return None
        
        try:
            # Prepare data
            data = np.expand_dims(np.array(list(self.buffer)), axis=0).astype(np.float32)
            
            # Execute prediction
            self.interpreter.set_tensor(self.input_details[0]['index'], data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process results
            fall_prob = float(output.flatten()[0])
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {'prediction': prediction, 'probability': fall_prob}
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

def get_current_timestamp():
    """Current KST time"""
    return datetime.now(KST).isoformat()

def create_imu_package(data, user_id):
    """Create IMU data package"""
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
    """Create fall data package"""
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
    """WebSocket connection management"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    retry_delay = 1
    
    while True:
        try:
            print(f"WebSocket connection attempt: {url}")
            async with websockets.connect(url) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                retry_delay = 1
                print("✅ WebSocket connected successfully")
                
                # Start data transmission loop
                await data_sender.send_loop()
                
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
        
        print(f"Waiting for reconnection: {retry_delay}s")
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)

# IMU 송신 버퍼 (100Hz로 쌓고 10Hz로 송신)
imu_send_buffer = deque(maxlen=SAMPLING_RATE)  # 1초치 버퍼

def main():
    """Main function"""
    print("🚀 Fall Detection System Starting")
    print(f"Current time (KST): {get_current_timestamp()}")
    
    # Initialize
    try:
        sensor = SimpleSensor()
        detector = SimpleFallDetector()
        data_sender = SafeDataSender()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Exit handler
    def signal_handler(sig, frame):
        print("\nShutting down program...")
        if not data_sender.fall_queue.empty():
            print(f"Remaining fall data: {data_sender.fall_queue.qsize()} items")
            time.sleep(3)
        print("Program terminated")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start WebSocket client
    def start_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(data_sender))
    
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    
    print("🔄 Data collection started")
    
    # Fill initial buffer
    for _ in range(SEQ_LENGTH):
        data = sensor.get_data()
        detector.add_data(data)
        imu_send_buffer.append(data)  # Add to buffer only
        time.sleep(1.0 / SAMPLING_RATE)
    
    print("🎯 Fall detection started")
    
    # --- IMU transmission loop (10Hz) ---
    def imu_send_loop():
        while True:
            if imu_send_buffer:
                latest_data = imu_send_buffer[-1]
                data_sender.add_imu_data(create_imu_package(latest_data, USER_ID))
            time.sleep(1.0 / SEND_RATE)
    
    imu_sender_thread = threading.Thread(target=imu_send_loop, daemon=True)
    imu_sender_thread.start()
    
    # --- Main loop (100Hz) ---
    last_print = time.time()
    
    while True:
        try:
            data = sensor.get_data()
            detector.add_data(data)
            imu_send_buffer.append(data)  # Add to buffer at 100Hz only
            
            # Debug output (every 30 seconds to reduce frequency)
            current_time = time.time()
            if current_time - last_print >= 30.0:
                print(f"Connection: {'✅ Connected' if data_sender.connected else '❌ Disconnected'}")
                print(f"Sampling: {SAMPLING_RATE}Hz, Prediction interval: {STRIDE} samples")
                last_print = current_time
            
            # Fall prediction
            if detector.should_predict():
                result = detector.predict()
                if result:
                    # Output prediction probability every stride (exclude coordinate info)
                    probability = result['probability']
                    prediction_text = 'Fall' if result['prediction'] == 1 else 'Normal'
                    
                    # Color indication based on probability
                    if probability >= 0.8:
                        print(f"🔴 Prediction: {probability:.4f} ({prediction_text}) - Danger")
                    elif probability >= 0.5:
                        print(f"🟡 Prediction: {probability:.4f} ({prediction_text}) - Warning")
                    else:
                        print(f"🟢 Prediction: {probability:.4f} ({prediction_text}) - Safe")
                    
                    if result['prediction'] == 1:
                        print(f"\n🚨🚨🚨 FALL DETECTED! Confidence: {probability:.2%} 🚨🚨🚨")
                        fall_package = create_fall_package(USER_ID, probability, data)
                        data_sender.add_fall_data(fall_package)
                        print("🚨 FALL ALERT!")
                        time.sleep(2)
            
            time.sleep(1.0 / SAMPLING_RATE)
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()