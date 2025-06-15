#!/usr/bin/env python3
"""
실시간 보행 감지 시스템
MPU-6050 센서를 이용한 허리 부착형 gait/non-gait 분류기
"""

import numpy as np
import scipy.signal as sg
import time
import collections
from typing import Tuple, Optional
import smbus2

class MPU6050:
    """MPU-6050 센서 인터페이스"""
    
    def __init__(self, bus_num=1, device_addr=0x68):
        self.bus = smbus2.SMBus(bus_num)
        self.addr = device_addr
        self._initialize_sensor()
    
    def _initialize_sensor(self):
        """센서 초기화"""
        # Power management 레지스터 설정 (wake up)
        self.bus.write_byte_data(self.addr, 0x6B, 0x00)
        
        # 가속도계 범위 설정 (±2g)
        self.bus.write_byte_data(self.addr, 0x1C, 0x00)
        
        # 자이로스코프 범위 설정 (±250°/s)
        self.bus.write_byte_data(self.addr, 0x1B, 0x00)
        
        time.sleep(0.1)
    
    def read_accel(self) -> np.ndarray:
        """가속도 데이터 읽기 (g 단위)"""
        # 가속도 레지스터 읽기 (0x3B~0x40)
        data = self.bus.read_i2c_block_data(self.addr, 0x3B, 6)
        
        # 16비트 데이터 변환
        ax = self._convert_raw_data(data[0], data[1])
        ay = self._convert_raw_data(data[2], data[3])
        az = self._convert_raw_data(data[4], data[5])
        
        # g 단위로 변환 (±2g 범위)
        scale = 16384.0  # LSB/g for ±2g range
        return np.array([ax/scale, ay/scale, az/scale])
    
    def _convert_raw_data(self, high_byte, low_byte):
        """16비트 signed 데이터 변환"""
        value = (high_byte << 8) | low_byte
        if value > 32767:
            value -= 65536
        return value

class GaitDetector:
    """실시간 보행 감지 시스템"""
    
    def __init__(self, sampling_rate=30):
        self.FS = sampling_rate
        self.WIN = int(2 * self.FS)  # 2초 윈도우 (60프레임)
        
        # 알고리즘 파라미터
        self.global_thr = 0.05  # g
        self.peak_thr = 0.40    # g
        self.min_len_s = 2.0    # 초
        
        # 신호처리 필터
        self.lpf = sg.butter(4, 0.25/(self.FS/2), output='sos')
        
        # 데이터 버퍼
        self.buf_acc = collections.deque(maxlen=self.WIN)
        
        # 상태 관리
        self.current_state = "non-gait"  # 초기 상태
        self.frame_labels = collections.deque(maxlen=self.WIN)  # 각 프레임 라벨
        self.state_counter = 0  # 현재 상태 지속 프레임 수
        
        print(f"Gait detection system initialized (Sampling: {self.FS}Hz, Window: {self.WIN} frames)")
    
    def process_frame(self, accel_data: np.ndarray) -> Tuple[str, str, float]:
        """
        새로운 프레임 처리
        
        Args:
            accel_data: [ax, ay, az] 가속도 데이터 (g 단위)
            
        Returns:
            (current_state, frame_label, confidence): 현재 상태, 프레임 라벨, 신뢰도
        """
        # 버퍼에 데이터 추가
        self.buf_acc.append(accel_data.copy())
        
        # 윈도우가 채워지지 않은 경우
        if len(self.buf_acc) < self.WIN:
            self.frame_labels.append("unknown")
            return self.current_state, "unknown", 0.0
        
        # 윈도우 분석
        frame_label, confidence = self._analyze_window()
        
        # 중앙 프레임 라벨링 (1초 지연)
        center_idx = self.WIN // 2
        if len(self.frame_labels) >= center_idx:
            # 현재 분석 결과를 중앙 프레임에 할당
            self.frame_labels.append(frame_label)
        else:
            self.frame_labels.append("unknown")
        
        # 상태 변화 로직
        self._update_state()
        
        return self.current_state, frame_label, confidence
    
    def _analyze_window(self) -> Tuple[str, float]:
        """윈도우 분석하여 프레임 라벨 결정"""
        acc = np.vstack(self.buf_acc)
        
        # ① 중력 분리 (4차 Butterworth LPF 0.25Hz)
        g_est = sg.sosfiltfilt(self.lpf, acc, axis=0)
        dyn = acc - g_est
        
        # ② 크기 신호 생성
        vm = np.linalg.norm(dyn, axis=1)
        
        # ③ 전역 저강도 필터
        I_act = (vm > self.global_thr).astype(int)
        
        # ④ 활동 스무딩 (Gaussian 1s 커널)
        gaussian_kernel = sg.windows.gaussian(self.FS, std=self.FS//2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        L_act = np.convolve(I_act, gaussian_kernel, mode='same')
        
        # ⑤ Gap 병합 (간단화 버전)
        # 연속 활동 검사
        if L_act.mean() < 0.5:
            return "non-gait", 0.1
        
        # ⑥-⑦ 보행 검증
        confidence = self._verify_gait(vm)
        
        if confidence > 0.5:
            return "gait", confidence
        else:
            return "non-gait", 1.0 - confidence
    
    def _verify_gait(self, vm: np.ndarray) -> float:
        """보행 검증 및 신뢰도 계산"""
        confidence = 0.0
        
        # 조건 1: 0.4g 초과율 검사 (2.5% 이하여야 함)
        peak_ratio = (vm > self.peak_thr).mean()
        if peak_ratio <= 0.025:
            confidence += 0.4
        
        # 조건 2: 주파수 검사
        try:
            f, pxx = sg.welch(vm, self.FS, nperseg=min(self.WIN, 60))
            
            # 피크 주파수 찾기 (DC 제외)
            peak_idx = np.argmax(pxx[1:]) + 1
            f_peak = f[peak_idx]
            peak_power = pxx[peak_idx]
            total_power = pxx.sum()
            
            # 주파수 범위 검사 (0.8-3.0 Hz)
            if 0.8 <= f_peak <= 3.0:
                confidence += 0.3
            
            # 피크 파워 검사 (전체의 15% 이상)
            if peak_power >= 0.15 * total_power:
                confidence += 0.3
                
        except:
            # FFT 실패 시 신뢰도 감소
            confidence *= 0.5
        
        return confidence
    
    def _update_state(self):
        """상태 변화 로직"""
        if len(self.frame_labels) < 2:
            return
        
        current_frame_label = self.frame_labels[-1]
        
        # 동일한 라벨이면 카운터 증가
        if len(self.frame_labels) >= 2 and current_frame_label == self.frame_labels[-2]:
            self.state_counter += 1
        else:
            self.state_counter = 1
        
        # 상태 전환 확인 (2초 = 60프레임)
        if self.state_counter >= self.WIN:  # 2초 지속
            if current_frame_label == "gait" and self.current_state == "non-gait":
                self.current_state = "gait"
                print(f"State changed: non-gait → gait (frame {len(self.frame_labels)})")
            elif current_frame_label == "non-gait" and self.current_state == "gait":
                self.current_state = "non-gait"
                print(f"State changed: gait → non-gait (frame {len(self.frame_labels)})")

def main():
    """메인 실행 함수"""
    try:
        # 센서 및 감지기 초기화
        sensor = MPU6050()
        detector = GaitDetector(sampling_rate=30)
        
        print("Gait detection started... (Ctrl+C to exit)")
        print("Format: [state] frame_label (confidence) | acceleration_values")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.time()
            
            # 센서 데이터 읽기
            accel = sensor.read_accel()
            
            # 보행 감지 처리
            state, label, confidence = detector.process_frame(accel)
            
            # 결과 출력 (매 프레임마다)
            elapsed = time.time() - start_time
            frame_time = frame_count * 0.033  # 30Hz = 0.033초/프레임
            print(f"[{state:8s}] {label:8s} ({confidence:.2f}) | "
                  f"a=({accel[0]:+.3f}, {accel[1]:+.3f}, {accel[2]:+.3f}) | "
                  f"f={frame_count:4d} t={frame_time:.2f}s | cnt={detector.state_counter}/60")
            
            frame_count += 1
            
            # 30Hz 유지를 위한 대기
            elapsed = time.time() - loop_start
            sleep_time = max(0, 1.0/30.0 - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nGait detection terminated")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()