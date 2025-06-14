#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이 실제 센서 데이터 웹소켓 전송 클라이언트
작성일: 2024-12-28
기능: MPU6050 IMU 센서 데이터를 웹소켓을 통해 서버로 실시간 전송
센서 연결: SDA -> GPIO 2 (Pin 3), SCL -> GPIO 3 (Pin 5), VCC -> 3.3V, GND -> GND
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
    print("센서 라이브러리를 찾을 수 없습니다. 시뮬레이션 모드로 실행됩니다.")
    print("실제 센서 사용을 위해 다음 명령을 실행하세요:")
    print("pip install -r requirements.txt")

class RealSensorClient:
    def __init__(self, server_host="172.30.1.93", server_port=8765, output_format="json"):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{server_host}:{server_port}"
        self.output_format = output_format  # "json" 또는 "csv"
        self.running = False
        self.frame_number = 0
        self.start_time = time.time()
        
        # CSV 헤더 정의
        self.csv_header = [
            "frame_number", "sync_timestamp", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 센서 초기화
        self.sensor_init()
    
    def sensor_init(self):
        """센서 초기화"""
        if SENSOR_AVAILABLE:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA)
                self.mpu = adafruit_mpu6050.MPU6050(self.i2c)
                self.logger.info("MPU6050 센서 초기화 완료")
            except Exception as e:
                self.logger.error(f"센서 초기화 실패: {e}")
                self.mpu = None
        else:
            self.mpu = None
    
    def get_sensor_data(self):
        """센서 데이터 읽기"""
        # 현재 시간 기준 타임스탬프 계산
        current_time = time.time()
        sync_timestamp = round((current_time - self.start_time), 3)
        
        if self.mpu is not None:
            try:
                # 실제 센서 데이터 읽기
                acceleration = self.mpu.acceleration
                gyro = self.mpu.gyro
                
                accel_x = round(acceleration[0], 3)
                accel_y = round(acceleration[1], 3)
                accel_z = round(acceleration[2], 3)
                gyro_x = round(gyro[0], 5)
                gyro_y = round(gyro[1], 5)
                gyro_z = round(gyro[2], 5)
                
            except Exception as e:
                self.logger.error(f"센서 데이터 읽기 실패: {e}")
                # 에러 시 기본값 사용
                accel_x, accel_y, accel_z = 0.0, 9.8, 0.0
                gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
        else:
            # 시뮬레이션 데이터
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
        """데이터를 지정된 형식으로 변환"""
        if self.output_format == "json":
            return json.dumps(sensor_data)
        elif self.output_format == "csv":
            csv_data = [
                sensor_data["frame_number"],
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
        """웹소켓을 통해 센서 데이터 전송"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.logger.info(f"웹소켓 서버에 연결됨: {self.server_url}")
                self.logger.info(f"출력 형식: {self.output_format}")
                self.running = True
                
                # CSV 형식인 경우 헤더 전송
                if self.output_format == "csv":
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(self.csv_header)
                    header_line = output.getvalue().strip()
                    await websocket.send(header_line)
                    self.logger.info("CSV 헤더 전송 완료")
                
                while self.running:
                    try:
                        # 센서 데이터 읽기
                        sensor_data = self.get_sensor_data()
                        
                        # 데이터 형식 변환 및 전송
                        formatted_data = self.format_data(sensor_data)
                        await websocket.send(formatted_data)
                        
                        if self.frame_number % 30 == 0:  # 30프레임마다 로그 출력
                            self.logger.info(f"Frame {self.frame_number}: 데이터 전송 중...")
                        
                        # 프레임 번호 증가
                        self.frame_number += 1
                        
                        # 30Hz 주기로 전송 (약 33ms 간격)
                        await asyncio.sleep(0.033)
                        
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("웹소켓 연결이 종료되었습니다.")
                        break
                    except Exception as e:
                        self.logger.error(f"데이터 전송 중 오류: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            self.logger.error(f"웹소켓 연결 실패: {e}")
    
    def stop(self):
        """데이터 전송 중지"""
        self.running = False
        self.logger.info("센서 데이터 전송을 중지합니다.")

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 처리
    output_format = "json"
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["csv", "json"]:
            output_format = sys.argv[1].lower()
    
    print(f"라즈베리파이 센서 데이터 전송 클라이언트")
    print(f"서버 주소: 172.30.1.93:8765")
    print(f"출력 형식: {output_format.upper()}")
    print(f"종료하려면 Ctrl+C를 누르세요.")
    print("-" * 50)
    
    client = RealSensorClient(output_format=output_format)
    
    try:
        asyncio.run(client.send_sensor_data())
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다...")
        client.stop()

if __name__ == "__main__":
    main() 