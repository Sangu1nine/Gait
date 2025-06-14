from smbus2 import SMBus
from bitstring import Bits
import math
import time
import pandas as pd
import datetime
import numpy as np
import socket
import json
import threading
 
bus = SMBus(1)
DEV_ADDR = 0x68
 
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0
 
register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0

# WiFi 통신 설정
#WIFI_SERVER_IP = '192.168.0.186' 
WIFI_SERVER_IP = '172.20.10.12'  # 로컬 PC의 IP 주소 (수정됨)
WIFI_SERVER_PORT = 5000  # 통신 포트
wifi_client = None
wifi_connected = False
send_data_queue = []
 
def read_data(register):
    high = bus.read_byte_data(DEV_ADDR,register)
    low = bus.read_byte_data(DEV_ADDR,register+1)
    val = (high << 8) + low
    return val
 
def twocomplements(val):
    s = Bits(uint=val,length=16)
    return s.int
 
def gyro_dps(val):
    return twocomplements(val)/sensitive_gyro
 
def accel_g(val):
    return twocomplements(val)/sensitive_accel
 
def dist(a,b):
    return math.sqrt((a*a)+(b*b))
 
def get_x_rotation(x,y,z):
    radians = math.atan(x/dist(y,z))
    return radians
 
def get_y_rotation(x,y,z):
    radians = math.atan(y/dist(x,z))
    return radians

def connect_wifi():
    global wifi_client, wifi_connected
    try:
        wifi_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        wifi_client.connect((WIFI_SERVER_IP, WIFI_SERVER_PORT))
        wifi_connected = True
        print(f"WiFi connection successful: {WIFI_SERVER_IP}:{WIFI_SERVER_PORT}")
        return True
    except Exception as e:
        print(f"WiFi connection failed: {str(e)}")
        wifi_connected = False
        return False

def send_data_thread():
    global send_data_queue, wifi_client, wifi_connected
    
    while wifi_connected:
        if len(send_data_queue) > 0:
            try:
                # 큐에서 데이터 가져오기
                sensor_data = send_data_queue.pop(0)
                # JSON 형식으로 변환하여 전송
                data_json = json.dumps(sensor_data)
                wifi_client.sendall((data_json + '\n').encode('utf-8'))
            except Exception as e:
                print(f"Data transmission error: {str(e)}")
                wifi_connected = False
                break
        else:
            time.sleep(0.001)  # 큐가 비어있을 때 CPU 사용량 줄이기

def close_wifi():
    global wifi_client, wifi_connected
    if wifi_client:
        try:
            wifi_client.close()
            print("WiFi connection closed")
        except:
            pass
    wifi_connected = False
 
# 센서 초기화
bus.write_byte_data(DEV_ADDR,0x6B,0b00000000)

# 데이터 프레임 준비
columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
data = []

# 파일명 설정 (현재 시간 기반)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"imu_data_{timestamp}.csv"

print(f"Data collection started (30Hz) - Save file: {filename}")
print("Press Ctrl+C to stop collection")

# WiFi 연결 시도
wifi_thread = None
if connect_wifi():
    # 데이터 전송 스레드 시작
    wifi_thread = threading.Thread(target=send_data_thread)
    wifi_thread.daemon = True
    wifi_thread.start()

# 초기 시간
start_time = time.time()
sample_count = 0
target_hz = 30  # 목표 샘플링 레이트

try:
    while True:
        # 현재 샘플 시간 계산
        current_time = time.time()
        elapsed = current_time - start_time
        
        # 가속도 데이터 읽기
        accel_x = accel_g(read_data(register_accel_xout_h))
        accel_y = accel_g(read_data(register_accel_yout_h))
        accel_z = accel_g(read_data(register_accel_zout_h))
        
        # 자이로스코프 데이터 읽기
        gyro_x = gyro_dps(read_data(register_gyro_xout_h))
        gyro_y = gyro_dps(read_data(register_gyro_yout_h))
        gyro_z = gyro_dps(read_data(register_gyro_zout_h))
        
        # 데이터 추가
        data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        sample_count += 1
        
        # WiFi로 데이터 전송 (연결된 경우)
        if wifi_connected:
            sensor_data = {
                'timestamp': elapsed,
                'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
                'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z}
            }
            send_data_queue.append(sensor_data)
        
        # 1초에 한 번씩 진행 상황 출력
        if sample_count % 30 == 0:
            print(f"Samples: {sample_count}, Elapsed time: {elapsed:.2f}s, Sampling rate: {sample_count/elapsed:.2f}Hz")
            print(f"Acceleration(g): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
            print(f"Gyroscope(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
            if wifi_connected:
                print(f"WiFi transmission status: Connected (Queue length: {len(send_data_queue)})")
            else:
                print("WiFi transmission status: Not connected")
        
        # 샘플링 레이트 유지 (30Hz = 0.033초 간격)
        next_sample_time = start_time + (sample_count * (1.0 / target_hz))
        sleep_time = next_sample_time - time.time()
        
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nData collection stopped!")
    
    # 데이터프레임 생성
    df = pd.DataFrame(data, columns=columns)
    
    # 인덱스 설정 (시간 기반)
    df.index = np.arange(len(df)) / target_hz
    df.index.name = 'Time(s)'
    
    # CSV 파일로 저장
    df.to_csv(filename)
    print(f"Data saved: {filename} (Total {len(df)} samples)")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
finally:
    # WiFi 연결 종료
    close_wifi()
    
    bus.close()
    print("I2C bus closed") 