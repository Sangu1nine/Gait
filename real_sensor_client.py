#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Real Sensor Data WebSocket Transmission Client
Created: 2024-12-28
Modified: 2024-12-28 - Applied smbus2-based sensor reading method from get_data_100hz.py
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
    from smbus2 import SMBus
    from bitstring import Bits
    import math
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("Sensor libraries not found. Running in simulation mode.")
    print("To use real sensors, run the following command:")
    print("pip install smbus2 bitstring")

class RealSensorClient:
    def __init__(self, server_host="172.20.10.12", server_port=8765, output_format="json"):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{server_host}:{server_port}"
        self.output_format = output_format  # "json" 또는 "csv"
        self.running = False
        self.frame_number = 0
        self.start_time = time.time()
        
        # MPU6050 센서 설정 (get_data_100hz.py에서 가져옴)
        self.DEV_ADDR = 0x68
        
        # 자이로스코프 레지스터
        self.register_gyro_xout_h = 0x43
        self.register_gyro_yout_h = 0x45
        self.register_gyro_zout_h = 0x47
        self.sensitive_gyro = 131.0
        
        # 가속도계 레지스터
        self.register_accel_xout_h = 0x3B
        self.register_accel_yout_h = 0x3D
        self.register_accel_zout_h = 0x3F
        self.sensitive_accel = 16384.0
        
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
                # I2C 버스 속도를 낮춰서 안정성 향상
                self.bus = SMBus(1)
                
                # 센서 초기화 (get_data_100hz.py와 동일한 방식)
                self.bus.write_byte_data(self.DEV_ADDR, 0x6B, 0b00000000)
                time.sleep(0.1)  # 센서 초기화 대기
                
                # 추가 센서 설정 (안정성 향상)
                self.bus.write_byte_data(self.DEV_ADDR, 0x1A, 0x00)  # CONFIG 레지스터 - 필터 비활성화
                self.bus.write_byte_data(self.DEV_ADDR, 0x1B, 0x00)  # GYRO_CONFIG - ±250°/s
                self.bus.write_byte_data(self.DEV_ADDR, 0x1C, 0x00)  # ACCEL_CONFIG - ±2g
                time.sleep(0.05)  # 추가 설정 대기
                
                # 센서 연결 테스트
                test_val = self.bus.read_byte_data(self.DEV_ADDR, 0x75)  # WHO_AM_I 레지스터
                if test_val == 0x68:
                    self.logger.info("MPU6050 sensor initialization completed (smbus2)")
                else:
                    self.logger.warning(f"Sensor WHO_AM_I test: expected 0x68, got 0x{test_val:02x}")
                    
            except Exception as e:
                self.logger.error(f"Sensor initialization failed: {e}")
                self.bus = None
        else:
            self.bus = None
    
    def read_data(self, register, retry_count=3):
        """레지스터에서 16비트 데이터 읽기 (재시도 로직 포함)"""
        for attempt in range(retry_count):
            try:
                # 연속 레지스터 읽기 사이에 짧은 지연 추가
                high = self.bus.read_byte_data(self.DEV_ADDR, register)
                time.sleep(0.0001)  # 0.1ms 지연
                low = self.bus.read_byte_data(self.DEV_ADDR, register + 1)
                val = (high << 8) + low
                return val
            except Exception as e:
                if attempt == retry_count - 1:  # 마지막 시도
                    raise e
                time.sleep(0.002)  # 2ms 대기 후 재시도
    
    def twocomplements(self, val):
        """2의 보수 변환 (get_data_100hz.py에서 가져옴)"""
        s = Bits(uint=val, length=16)
        return s.int
    
    def gyro_dps(self, val):
        """자이로스코프 데이터를 degree/second로 변환 (get_data_100hz.py에서 가져옴)"""
        return self.twocomplements(val) / self.sensitive_gyro
    
    def accel_g(self, val):
        """가속도 데이터를 g로 변환 (get_data_100hz.py에서 가져옴)"""
        return self.twocomplements(val) / self.sensitive_accel
    
    def get_sensor_data(self):
        """Read sensor data"""
        # Calculate timestamp based on frame number for consistent 30Hz timing
        sync_timestamp = round(self.frame_number * 0.033, 3)
        
        if self.bus is not None:
            max_retries = 3  # 재시도 횟수 증가
            for retry in range(max_retries):
                try:
                    # 센서 데이터 읽기 사이에 짧은 지연 추가 (안정성 향상)
                    accel_x = round(self.accel_g(self.read_data(self.register_accel_xout_h)), 3)
                    time.sleep(0.0005)  # 0.5ms 지연
                    accel_y = round(self.accel_g(self.read_data(self.register_accel_yout_h)), 3)
                    time.sleep(0.0005)
                    accel_z = round(self.accel_g(self.read_data(self.register_accel_zout_h)), 3)
                    time.sleep(0.0005)
                    gyro_x = round(self.gyro_dps(self.read_data(self.register_gyro_xout_h)), 5)
                    time.sleep(0.0005)
                    gyro_y = round(self.gyro_dps(self.read_data(self.register_gyro_yout_h)), 5)
                    time.sleep(0.0005)
                    gyro_z = round(self.gyro_dps(self.read_data(self.register_gyro_zout_h)), 5)
                    break  # 성공하면 루프 종료
                    
                except Exception as e:
                    if retry == max_retries - 1:  # 마지막 재시도
                        # 에러 빈도를 줄이기 위해 10번에 1번만 로그 출력
                        if self.frame_number % 10 == 0:
                            self.logger.error(f"Failed to read sensor data after {max_retries} attempts: {e}")
                        # Use default values on error
                        accel_x, accel_y, accel_z = 0.0, 9.8, 0.0
                        gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
                    else:
                        time.sleep(0.01)  # 10ms 대기 후 재시도
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
        if self.bus:
            self.bus.close()
            self.logger.info("I2C bus closed")
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
    print(f"Sensor Interface: smbus2 (Direct register access)")
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