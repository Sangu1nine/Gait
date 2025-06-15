#!/usr/bin/env python3
"""
Real-time Gait Detection System - Raspberry Pi 4
30Hz IMU sensor, 60 window, 1 stride

MODIFIED [2024-12-19]: Initial implementation - Real-time IMU data collection and gait detection
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
        """Initialize IMU sensor (MPU6050)"""
        try:
            # Initialize I2C interface
            i2c = busio.I2C(board.SCL, board.SDA)
            self.mpu = adafruit_mpu6050.MPU6050(i2c)
            
            # Configure sensor settings
            self.mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_4_G
            self.mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_500_DPS
            
            print("✅ MPU6050 sensor initialized successfully")
            
        except Exception as e:
            print(f"❌ IMU sensor initialization failed: {e}")
            print("💡 Switching to simulation mode.")
            self.mpu = None
    
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
            
            # Load StandardScaler
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✅ StandardScaler loaded successfully: {scaler_path}")
            
            # Load label encoder
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded successfully: {label_encoder_path}")
            
            # Load optimal thresholds
            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)
            
            # Calculate threshold average
            threshold_values = list(thresholds.values())
            self.optimal_threshold = np.mean(threshold_values)
            print(f"✅ Optimal thresholds loaded successfully: {self.optimal_threshold:.3f}")
            print(f"📋 Fold-wise thresholds: {thresholds}")
            
        except Exception as e:
            print(f"❌ Failed to load model/preprocessing objects: {e}")
            raise
    
    def read_imu_data(self):
        """Read data from IMU sensor"""
        if self.mpu is None:
            # Simulation data (for testing)
            accel = [np.random.normal(0, 1) for _ in range(3)]
            gyro = [np.random.normal(0, 1) for _ in range(3)]
        else:
            # Real sensor data
            accel = list(self.mpu.acceleration)  # m/s²
            gyro = list(self.mpu.gyro)          # rad/s
        
        return accel + gyro  # Return 6 features
    
    def preprocess_data(self, window_data):
        """Data preprocessing (refer to README.md)"""
        # Convert to numpy array (1, 60, 6)
        X = np.array(window_data).reshape(1, self.window_size, self.n_features)
        
        # 3D -> 2D conversion
        X_2d = X.reshape(-1, self.n_features)
        
        # Apply StandardScaler (transform only, no fit!)
        X_scaled = self.scaler.transform(X_2d)
        
        # 2D -> 3D conversion
        X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled.astype(np.float32)
    
    def predict_tflite(self, X_preprocessed):
        """Predict using TFLite model"""
        # Set input data
        self.interpreter.set_tensor(self.input_details[0]['index'], X_preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def collect_data_continuously(self):
        """Continuously collect IMU data (30Hz)"""
        interval = 1.0 / self.sampling_rate  # 30Hz = 33.33ms interval
        
        while self.is_collecting:
            start_time = time.time()
            
            # Read IMU data
            sensor_data = self.read_imu_data()
            
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
            
            # Preprocessing
            X_preprocessed = self.preprocess_data(window_data)
            
            # Prediction
            y_prob = self.predict_tflite(X_preprocessed)
            
            # Apply threshold
            y_pred = (y_prob > self.optimal_threshold).astype(int)
            
            # Decode labels
            prediction_label = self.label_encoder.inverse_transform(y_pred.flatten())[0]
            confidence = y_prob[0][0]
            
            # Display results
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            status_icon = "🚶" if prediction_label == "gait" else "🧍"
            
            print(f"[{timestamp}] {status_icon} Prediction: {prediction_label} "
                  f"(confidence: {confidence:.3f}, threshold: {self.optimal_threshold:.3f})")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    def start_detection(self):
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
            # Keep main thread alive
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
    """Main function"""
    print("🤖 Raspberry Pi 4 Real-time Gait Detection System")
    print("=" * 50)
    
    try:
        # Initialize gait detection system
        detector = RealTimeGaitDetector()
        
        # Start real-time detection
        detector.start_detection()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Please check if model files and preprocessing object files are in the correct paths.")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Exiting program.")

if __name__ == "__main__":
    main() 