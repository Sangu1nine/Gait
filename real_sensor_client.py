#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Real Sensor Data WebSocket Transmission Client
Created: 2024-12-28
Function: Real-time transmission of MPU6050 IMU sensor data to server via WebSocket
Sensor Connection: SDA -> GPIO 2 (Pin 3), SCL -> GPIO 3 (Pin 5), VCC -> 3.3V, GND -> GND
"""

import asyncio
import websockets
import json
import csv
import io
import time
import logging
from datetime import datetime

try:
    import board
    import busio
    import adafruit_mpu6050
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("Sensor libraries not found. Running in simulation mode.")
    print("To use real sensors, run the following command:")
    print("pip install -r requirements.txt")

class RealSensorClient:
    def __init__(self, server_host="172.20.10.12", server_port=8765, output_format="json"):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{server_host}:{server_port}"
        self.output_format = output_format  # "json" 또는 "csv"
        self.running = False
        self.frame_number = 0
        self.start_time = time.time()
        
        # CSV header definition
        self.csv_header = [
            "frame_number", "sync_timestamp", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize sensor
        self.sensor_init()
    
    def sensor_init(self):
        """Initialize sensor"""
        if SENSOR_AVAILABLE:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA)
                self.mpu = adafruit_mpu6050.MPU6050(self.i2c)
                self.logger.info("MPU6050 sensor initialization completed")
            except Exception as e:
                self.logger.error(f"Sensor initialization failed: {e}")
                self.mpu = None
        else:
            self.mpu = None
    
    def get_sensor_data(self):
        """Read sensor data"""
        # Calculate timestamp based on frame number for consistent 30Hz timing
        sync_timestamp = round(self.frame_number * 0.033, 3)
        
        if self.mpu is not None:
            try:
                # Read actual sensor data
                acceleration = self.mpu.acceleration
                gyro = self.mpu.gyro
                
                accel_x = round(acceleration[0], 3)
                accel_y = round(acceleration[1], 3)
                accel_z = round(acceleration[2], 3)
                gyro_x = round(gyro[0], 5)
                gyro_y = round(gyro[1], 5)
                gyro_z = round(gyro[2], 5)
                
            except Exception as e:
                self.logger.error(f"Failed to read sensor data: {e}")
                # Use default values on error
                accel_x, accel_y, accel_z = 0.0, 9.8, 0.0
                gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
        else:
            # Simulation data
            import random
            accel_x = round(random.uniform(-2.0, 2.0), 3)
            accel_y = round(random.uniform(8.0, 12.0), 3)
            accel_z = round(random.uniform(0.5, 4.0), 3)
            gyro_x = round(random.uniform(-50.0, 50.0), 5)
            gyro_y = round(random.uniform(-50.0, 50.0), 5)
            gyro_z = round(random.uniform(-50.0, 50.0), 5)
        
        return {
            "frame_number": self.frame_number,
            "sync_timestamp": sync_timestamp,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z
        }
    
    def format_data(self, sensor_data):
        """Convert data to specified format"""
        if self.output_format == "json":
            return json.dumps(sensor_data)
        elif self.output_format == "csv":
            csv_data = [
                str(sensor_data["frame_number"]),
                f"{sensor_data['sync_timestamp']:.3f}",
                f"{sensor_data['accel_x']:.3f}",
                f"{sensor_data['accel_y']:.3f}",
                f"{sensor_data['accel_z']:.3f}",
                f"{sensor_data['gyro_x']:.5f}",
                f"{sensor_data['gyro_y']:.5f}",
                f"{sensor_data['gyro_z']:.5f}"
            ]
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(csv_data)
            return output.getvalue().strip()
    
    async def send_sensor_data(self):
        """Send sensor data via WebSocket"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.logger.info(f"Connected to WebSocket server: {self.server_url}")
                self.logger.info(f"Output format: {self.output_format}")
                self.running = True
                
                # Send header for CSV format
                if self.output_format == "csv":
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(self.csv_header)
                    header_line = output.getvalue().strip()
                    await websocket.send(header_line)
                    self.logger.info("CSV header sent successfully")
                
                while self.running:
                    try:
                        # Read sensor data
                        sensor_data = self.get_sensor_data()
                        
                        # Format and send data
                        formatted_data = self.format_data(sensor_data)
                        await websocket.send(formatted_data)
                        
                        if self.frame_number % 30 == 0:  # Log every 30 frames
                            self.logger.info(f"Frame {self.frame_number}: Transmitting data...")
                        
                        # Increment frame number
                        self.frame_number += 1
                        
                        # Transmit at 30Hz (approximately 33ms interval)
                        await asyncio.sleep(0.033)
                        
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("WebSocket connection closed.")
                        break
                    except Exception as e:
                        self.logger.error(f"Error during data transmission: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
    
    def stop(self):
        """Stop data transmission"""
        self.running = False
        self.logger.info("Sensor data transmission stopped.")

def main():
    """Main execution function"""
    import sys
    
    # Process command line arguments
    output_format = "json"
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["csv", "json"]:
            output_format = sys.argv[1].lower()
    
    print(f"Raspberry Pi Sensor Data Transmission Client")
    print(f"Server Address: 172.20.10.12:8765")
    print(f"Output Format: {output_format.upper()}")
    print(f"Press Ctrl+C to exit.")
    print("-" * 50)
    
    client = RealSensorClient(output_format=output_format)
    
    try:
        asyncio.run(client.send_sensor_data())
    except KeyboardInterrupt:
        print("\nShutting down program...")
        client.stop()

if __name__ == "__main__":
    main() 