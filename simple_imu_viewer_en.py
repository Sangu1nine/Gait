"""
==================================================================
IMU 데이터 뷰어 (Simple IMU Data Viewer)
==================================================================

이 프로그램은 IMU(관성 측정 장치) 센서 데이터를 실시간으로 수신하고 
시각화하는 도구입니다.

기능:
1. TCP/IP 소켓 서버를 통해 IMU 센서 데이터 수신 (포트 5000)
2. 가속도계(X,Y,Z) 및 자이로스코프(X,Y,Z) 데이터 실시간 그래프 표시
3. 수신된 데이터를 CSV 파일로 저장 ('received_data' 폴더)
4. 그래프 이미지를 PNG 파일로 저장 ('saved_graphs' 폴더)
5. 낙상 감지 이벤트 표시 및 기록

사용 방법:
- 프로그램 실행: python simple_imu_viewer_en.py
- IMU 센서/클라이언트를 포트 5000에 연결하면 자동으로 데이터 수신 시작
- 데이터 형식: JSON {"timestamp": 시간, "accel": {"x":값, "y":값, "z":값}, "gyro": {"x":값, "y":값, "z":값}}
- 낙상 감지 형식: JSON {"event": "fall_detected", "timestamp": 시간, "probability": 확률}
- 키보드 단축키:
  * 'S' 키: 현재 그래프를 PNG 이미지로 저장
  * Ctrl+C: 프로그램 종료 (데이터와 그래프 자동 저장)

요구사항:
- Python 3.6 이상
- PyQtGraph, NumPy, PyQt5 라이브러리 필요

개발자: NAKSANG
최종 수정일: 2025년 5월 13일
==================================================================
"""

import socket
import json
import threading
import datetime
import time
import os
import sys
import signal
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

# Server settings
SERVER_IP = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'
GRAPH_FOLDER = 'saved_graphs'  # 그래프 저장 폴더
FALL_LOG_FILE = os.path.join(DATA_FOLDER, 'fall_events.csv')  # 낙상 이벤트 로그 파일

# Global variables
received_data = []
fall_events = []  # 낙상 감지 이벤트 저장
fall_event_queue = []  # 낙상 이벤트 UI 업데이트를 위한 큐
data_lock = threading.Lock()
is_running = False
start_time = time.time()  # 프로그램 시작 시간 저장

# Data buffers for visualization
buffer_size = 500  # Keep only the most recent 500 data points
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# Initialize with some data to prevent empty graphs
for i in range(5):
    time_buffer.append(i)
    accel_x_buffer.append(0)
    accel_y_buffer.append(0)
    accel_z_buffer.append(0)
    gyro_x_buffer.append(0)
    gyro_y_buffer.append(0)
    gyro_z_buffer.append(0)

# Socket and UI variables
server_socket = None
app = None
win = None
plots = {}
curves = {}
fall_markers = []  # 낙상 이벤트 마커 저장

# Create data folder
def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Data folder created: {DATA_FOLDER}")
    
    # 그래프 저장 폴더 생성
    if not os.path.exists(GRAPH_FOLDER):
        os.makedirs(GRAPH_FOLDER)
        print(f"Graph folder created: {GRAPH_FOLDER}")
    
    # 낙상 이벤트 로그 파일 생성 (없는 경우)
    if not os.path.exists(FALL_LOG_FILE):
        with open(FALL_LOG_FILE, 'w') as f:
            f.write("Timestamp,Date_Time,Probability\n")
        print(f"Fall event log file created: {FALL_LOG_FILE}")

# Save received data to CSV
def save_data():
    global received_data, fall_events
    
    with data_lock:
        if received_data:
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_FOLDER, f"imu_data_{timestamp}.csv")
            
            # Save to CSV
            with open(filename, 'w') as f:
                f.write("Time,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n")
                for data in received_data:
                    f.write(f"{','.join(str(x) for x in data)}\n")
            
            print(f"Data saved: {filename} (Total: {len(received_data)} samples)")
            received_data = []
        
        # 낙상 이벤트 저장
        if fall_events:
            with open(FALL_LOG_FILE, 'a') as f:
                for event in fall_events:
                    timestamp, prob = event
                    date_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"{timestamp},{date_time},{prob}\n")
            
            print(f"Fall events saved: {len(fall_events)} events to {FALL_LOG_FILE}")
            fall_events = []

# Process fall detection event
def process_fall_event(event_data):
    global fall_events
    
    timestamp = event_data['timestamp']
    probability = event_data['probability']
    
    # 화면에 알림
    print(f"\n{'*' * 30}")
    print(f"낙상 감지! 시간: {datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')}")
    print(f"확률: {probability:.2%}")
    print(f"{'*' * 30}\n")
    
    # 이벤트 저장
    with data_lock:
        fall_events.append([timestamp, probability])
    
    # 그래프에 마커 추가
    add_fall_marker(timestamp)

# Add fall marker to graph
def add_fall_marker(timestamp):
    global fall_markers, plots
    
    # 가속도 그래프에 세로선 추가
    line_pen = pg.mkPen(color=(255, 0, 0), width=2, style=QtCore.Qt.DashLine)
    acc_line = pg.InfiniteLine(pos=timestamp, angle=90, pen=line_pen, movable=False)
    gyr_line = pg.InfiniteLine(pos=timestamp, angle=90, pen=line_pen, movable=False)
    
    # 텍스트 추가 (고정된 X,Y 위치에 표시)
    time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    
    # 낙상 이벤트 정보 저장 (메인 스레드에서 UI 업데이트를 위해)
    with data_lock:
        # 시간 정보와 함께 낙상 이벤트 큐에 추가
        fall_event_queue.append(time_str)
    
    # 그래프에는 선만 표시 (UI 요소는 아님)
    fall_markers.append((acc_line, gyr_line))
    plots['accel'].addItem(acc_line)
    plots['gyro'].addItem(gyr_line)

# Client connection handler
def handle_client(client_socket, address):
    global received_data, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer, start_time
    
    print(f"Client connected: {address}")
    buffer = ""
    
    try:
        while is_running:
            data = client_socket.recv(4096)
            if not data:
                break
                
            # Add to buffer
            buffer += data.decode('utf-8')
            
            # Process complete JSON objects
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                try:
                    # Parse JSON
                    data_json = json.loads(line)
                    
                    # 낙상 감지 이벤트인 경우
                    if 'event' in data_json and data_json['event'] == 'fall_detected':
                        process_fall_event(data_json)
                        continue
                    
                    # 일반 센서 데이터인 경우
                    if 'accel' in data_json and 'gyro' in data_json:
                        # Extract data
                        # get_data.py에서 elapsed 시간을 보내므로 이를 실제 타임스탬프로 변환
                        elapsed_time = data_json['timestamp']
                        timestamp = start_time + elapsed_time  # 실제 유닉스 타임스탬프로 변환
                        
                        accel_x = data_json['accel']['x']
                        accel_y = data_json['accel']['y']
                        accel_z = data_json['accel']['z']
                        gyro_x = data_json['gyro']['x']
                        gyro_y = data_json['gyro']['y']
                        gyro_z = data_json['gyro']['z']
                        
                        # Store data
                        with data_lock:
                            received_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                            
                            # Add to visualization buffers
                            time_buffer.append(timestamp)
                            accel_x_buffer.append(accel_x)
                            accel_y_buffer.append(accel_y)
                            accel_z_buffer.append(accel_z)
                            gyro_x_buffer.append(gyro_x)
                            gyro_y_buffer.append(gyro_y)
                            gyro_z_buffer.append(gyro_z)
                
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
    
    except Exception as e:
        print(f"Client handler error: {e}")
    
    finally:
        client_socket.close()
        print(f"Client disconnected: {address}")

# Update plot function (called by timer)
def update_plots():
    global fall_event_queue
    
    with data_lock:
        t = list(time_buffer)
        ax = list(accel_x_buffer)
        ay = list(accel_y_buffer)
        az = list(accel_z_buffer)
        gx = list(gyro_x_buffer)
        gy = list(gyro_y_buffer)
        gz = list(gyro_z_buffer)
        
        # 낙상 이벤트 큐에서 이벤트 가져오기
        local_fall_events = list(fall_event_queue)
        fall_event_queue.clear()
    
    # 낙상 이벤트가 있으면 UI 업데이트 (메인 스레드에서 수행)
    if local_fall_events:
        for time_str in local_fall_events:
            # 낙상 이벤트 목록에 시간 추가
            fall_label = plots['fall_list']
            current_text = fall_label.text
            new_text = current_text + f"\n• {time_str} - 낙상 감지됨"
            fall_label.setText(new_text)
    
    if len(t) > 0:
        # 시간 형식 변환 - 유닉스 타임스탬프를 시:분:초.밀리초 형식으로 변환
        formatted_times = []
        for timestamp in t:
            dt = datetime.datetime.fromtimestamp(timestamp)
            formatted_times.append(dt.strftime('%H:%M:%S.%f')[:-3])  # 밀리초까지만 표시
        
        # 데이터 업데이트
        curves['accel_x'].setData(t, ax)  # X축은 원래 타임스탬프 유지 (내부 계산용)
        curves['accel_y'].setData(t, ay)
        curves['accel_z'].setData(t, az)
        curves['gyro_x'].setData(t, gx)
        curves['gyro_y'].setData(t, gy)
        curves['gyro_z'].setData(t, gz)
        
        # 자동 X축 범위 조정
        if len(t) > 1:
            x_min = max(min(t), t[-1] - 10)  # 최근 10초 데이터만 표시
            x_max = t[-1] + 0.5  # 약간의 여백 추가
            plots['accel'].setXRange(x_min, x_max, padding=0)
            
        # X축 눈금 형식 설정 (보기 쉽게 시간으로 표시)
        def format_time_axis(x):
            try:
                dt = datetime.datetime.fromtimestamp(x)
                return dt.strftime('%H:%M:%S')
            except:
                return ''
            
        # X축 눈금 형식 설정
        plots['accel'].getAxis('bottom').setTicks([[(t[i], format_time_axis(t[i])) for i in range(0, len(t), len(t)//5) if i < len(t)]])
        plots['gyro'].getAxis('bottom').setTicks([[(t[i], format_time_axis(t[i])) for i in range(0, len(t), len(t)//5) if i < len(t)]])

# Server thread function
def server_thread_func():
    global server_socket, is_running
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)  # Set timeout
        server_socket.listen(1)
        
        print(f"Server started: {SERVER_IP}:{SERVER_PORT}")
        
        while is_running:
            try:
                client_socket, client_address = server_socket.accept()
                client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except OSError:
                # Socket closed - expected during shutdown
                if is_running:
                    print("Server socket closed")
                break
            except Exception as e:
                if is_running:
                    print(f"Connection accept error: {e}")
                    time.sleep(0.5)
    
    except Exception as e:
        print(f"Server thread error: {e}")
    
    finally:
        if server_socket:
            try:
                server_socket.close()
            except:
                pass
            print("Server thread terminated")

# Save graphs to image file
def save_graphs():
    try:
        # 타임스탬프로 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(GRAPH_FOLDER, f"imu_graph_{timestamp}.png")
        
        # 화면에 출력된 그래프 영역 캡처
        exporter = pg.exporters.ImageExporter(win.centralWidget)
        exporter.parameters()['width'] = 1000  # 이미지 너비 설정
        exporter.export(filename)
        
        print(f"Graphs saved to image: {filename}")
    except Exception as e:
        print(f"Error saving graphs: {e}")
        # 대안으로 전체 윈도우 캡처 시도
        try:
            exporter = pg.exporters.ImageExporter(win.scene())
            exporter.export(filename)
            print(f"Graphs saved using alternative method: {filename}")
        except Exception as e2:
            print(f"Alternative save method also failed: {e2}")

# 키 입력 처리 클래스
class KeyPressWindow(pg.GraphicsLayoutWidget):
    def __init__(self, **kwargs):
        super(KeyPressWindow, self).__init__(**kwargs)
        
    def keyPressEvent(self, event):
        # 'S' 키를 누르면 그래프 저장
        if event.key() == QtCore.Qt.Key_S:
            save_graphs()
            print("Screenshot taken (pressed S key)")
        else:
            super(KeyPressWindow, self).keyPressEvent(event)

# Exit handler
def handle_exit():
    global is_running, server_socket
    
    print("\nExit requested - cleaning up...")
    is_running = False
    
    # Save data
    print("Saving data...")
    save_data()
    
    # Save graphs
    print("Saving graphs...")
    save_graphs()
    
    # Close socket
    if server_socket:
        try:
            server_socket.close()
            print("Server socket closed")
        except:
            pass
    
    print("Cleanup completed")

# Signal handler for Ctrl+C
def signal_handler(signum, frame):
    print("\nCtrl+C detected - shutting down gracefully...")
    handle_exit()
    if app:
        app.quit()
    sys.exit(0)

# Main function
def main():
    global app, win, plots, curves, is_running, start_time
    
    # Signal handler 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 프로그램 시작 시간 기록
    start_time = time.time()
    
    # Create folder
    create_data_folder()
    
    # Setup PyQtGraph
    app = QtWidgets.QApplication([])
    
    # 밝은 배경 테마 설정
    pg.setConfigOption('background', 'w')  # 배경색을 흰색으로 설정
    pg.setConfigOption('foreground', 'k')  # 전경색을 검은색으로 설정
    
    # Create window with key press handling
    win = KeyPressWindow(show=True, title="IMU Data Monitoring with Fall Detection")
    win.resize(1000, 700)
    
    # 낙상 이벤트 리스트를 표시할 레이블 추가
    plots['fall_list'] = pg.LabelItem(text="낙상 감지 기록:", color='#aa0000')
    win.addItem(plots['fall_list'], row=0, col=1, rowspan=2)
    
    # 단축키 안내 추가 (텍스트 색상 검은색으로 변경)
    shortcut_label = pg.LabelItem(text="Press 'S' to save graphs as image, Ctrl+C to exit", color='#000000')
    win.addItem(shortcut_label, row=3, col=0)
    
    # 낙상 감지 상태 레이블 추가 (텍스트 색상 유지)
    fall_status_label = pg.LabelItem(text="Fall Detection Status: Monitoring", color='#00aa00')
    win.addItem(fall_status_label, row=2, col=0)
    
    # Connect exit handler
    app.aboutToQuit.connect(handle_exit)
    
    # Accelerometer graph
    plots['accel'] = win.addPlot(row=0, col=0)
    plots['accel'].setTitle("Acceleration (g)")
    plots['accel'].setLabel('left', "Acceleration", "g")
    plots['accel'].setLabel('bottom', "Time", "")  # 단위 제거
    plots['accel'].addLegend()
    plots['accel'].showGrid(x=True, y=True)
    plots['accel'].setYRange(-1, 1)  # 가속도 그래프 Y축 고정 (-1g ~ +1g)
    plots['accel'].disableAutoRange(axis=pg.ViewBox.YAxis)  # Y축 자동 스케일링 비활성화
    
    # 시간 축 형식 설정 (추가)
    plots['accel'].getAxis('bottom').setStyle(tickTextOffset=15)  # 시간 레이블과 축 사이 간격 늘림
    
    # Gyroscope graph
    win.nextRow()
    plots['gyro'] = win.addPlot(row=1, col=0)
    plots['gyro'].setTitle("Gyroscope (°/s)")
    plots['gyro'].setLabel('left', "Angular Velocity", "°/s")
    plots['gyro'].setLabel('bottom', "Time", "")  # 단위 제거
    plots['gyro'].addLegend()
    plots['gyro'].showGrid(x=True, y=True)
    plots['gyro'].setYRange(-125, 125)  # 자이로스코프 그래프 Y축 고정 (-125°/s ~ +125°/s)
    plots['gyro'].disableAutoRange(axis=pg.ViewBox.YAxis)  # Y축 자동 스케일링 비활성화
    
    # 시간 축 형식 설정 (추가)
    plots['gyro'].getAxis('bottom').setStyle(tickTextOffset=15)  # 시간 레이블과 축 사이 간격 늘림
    
    # Link X axes for simultaneous scrolling
    plots['gyro'].setXLink(plots['accel'])
    
    # Create data curves
    curves['accel_x'] = plots['accel'].plot(pen=(255,0,0), name="X-axis")
    curves['accel_y'] = plots['accel'].plot(pen=(0,150,0), name="Y-axis")
    curves['accel_z'] = plots['accel'].plot(pen=(0,0,255), name="Z-axis")
    
    curves['gyro_x'] = plots['gyro'].plot(pen=(255,0,0), name="X-axis")
    curves['gyro_y'] = plots['gyro'].plot(pen=(0,150,0), name="Y-axis")
    curves['gyro_z'] = plots['gyro'].plot(pen=(0,0,255), name="Z-axis")
    
    # Update timer
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(50)  # Update every 50ms
    
    # Start server thread
    is_running = True
    server_thread = threading.Thread(target=server_thread_func)
    server_thread.daemon = True
    server_thread.start()
    
    # Start GUI event loop
    print("IMU Data Reception and Fall Detection Visualization Started")
    print("Press Ctrl+C to stop and save data")
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        handle_exit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTerminated by keyboard interrupt")
        handle_exit()
    except Exception as e:
        print(f"Unexpected error: {e}")
        handle_exit()
    finally:
        # Final cleanup
        is_running = False
        save_data() 