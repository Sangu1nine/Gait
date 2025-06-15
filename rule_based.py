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
    """
    실시간 보행 감지 시스템 (통합 설계)
    - Enhanced Butterworth filtering
    - Majority voting with adaptive thresholds  
    - Hysteresis-based state management
    - State-specific decision criteria
    """
    
    def __init__(self, sampling_rate=30):
        self.FS = sampling_rate
        
        # 통합 윈도우 설정
        self.analysis_window = int(2 * self.FS)     # 60프레임 (2초) - 기본 분석
        self.decision_window = int(3 * self.FS)     # 90프레임 (3초) - 의사결정
        
        # 알고리즘 파라미터 (중간 수준으로 조정)
        self.global_thr = 0.06  # g (0.07과 0.05의 중간)
        self.peak_thr = 0.40    # g
        
        # 통합 필터 시스템
        self.lpf_gravity = sg.butter(4, 0.25/(self.FS/2), output='sos')    # 중력 분리
        self.lpf_dynamic = sg.butter(4, 6.0/(self.FS/2), output='sos')     # 동적 가속도 노이즈 제거
        
        # 통합 상태별 임계값 (중간 수준으로 조정)
        self.thresholds = {
            'gait_start': 0.79,      # non-gait → gait: 79% (0.83과 0.75의 중간)
            'gait_maintain': 0.525,  # gait 유지: 52.5% (0.55와 0.50의 중간)
            'gait_end': 0.275,       # gait → non-gait: 27.5% (0.25와 0.30의 중간)
            'confidence_gait': 0.375, # gait 상태 신뢰도 (0.4와 0.35의 중간)
            'confidence_non_gait': 0.65  # non-gait 상태 신뢰도 (0.7과 0.6의 중간)
        }
        
        # 히스테리시스 설정
        self.hysteresis_frames = {
            'start': 60,    # 2초 - 보행 시작
            'end': 90       # 3초 - 보행 종료  
        }
        
        # 데이터 버퍼 (최대 윈도우 크기로 설정)
        self.buf_acc = collections.deque(maxlen=self.decision_window)
        
        # 상태 관리
        self.current_state = "non-gait"
        self.frame_labels = collections.deque(maxlen=self.decision_window)
        self.frame_confidences = collections.deque(maxlen=self.decision_window)
        
        # 디버깅 정보
        self.debug_info = {
            'gait_ratio': 0.0,
            'decision_basis': '',
            'frames_analyzed': 0
        }
        
        print(f"Enhanced Gait Detector initialized:")
        print(f"- Sampling: {self.FS}Hz")
        print(f"- Analysis window: {self.analysis_window} frames ({self.analysis_window/self.FS:.1f}s)")
        print(f"- Decision window: {self.decision_window} frames ({self.decision_window/self.FS:.1f}s)")
        print(f"- Hysteresis: start={self.hysteresis_frames['start']/self.FS:.1f}s, end={self.hysteresis_frames['end']/self.FS:.1f}s")
    
    def process_frame(self, accel_data: np.ndarray) -> Tuple[str, str, float]:
        """
        통합된 프레임 처리
        
        Args:
            accel_data: [ax, ay, az] 가속도 데이터 (g 단위)
            
        Returns:
            (current_state, frame_label, confidence): 현재 상태, 프레임 라벨, 신뢰도
        """
        # 버퍼에 데이터 추가
        self.buf_acc.append(accel_data.copy())
        
        # 최소 분석 윈도우가 채워지지 않은 경우
        if len(self.buf_acc) < self.analysis_window:
            self.frame_labels.append("unknown")
            self.frame_confidences.append(0.0)
            return self.current_state, "unknown", 0.0
        
        # 통합 윈도우 분석
        frame_label, confidence = self._enhanced_analyze_window()
        
        # 프레임 라벨 및 신뢰도 저장
        self.frame_labels.append(frame_label)
        self.frame_confidences.append(confidence)
        
        # 통합 상태 업데이트
        self._integrated_state_update()
        
        return self.current_state, frame_label, confidence
    
    def _enhanced_analyze_window(self) -> Tuple[str, float]:
        """향상된 윈도우 분석 (다중 필터 적용)"""
        acc = np.vstack(list(self.buf_acc)[-self.analysis_window:])
        
        # ① 이중 필터링
        # 중력 분리
        g_est = sg.sosfiltfilt(self.lpf_gravity, acc, axis=0)
        dyn = acc - g_est
        
        # 동적 가속도 노이즈 제거
        dyn_filtered = sg.sosfiltfilt(self.lpf_dynamic, dyn, axis=0)
        
        # ② 크기 신호 생성 (필터링된 신호 사용)
        vm = np.linalg.norm(dyn_filtered, axis=1)
        
        # ③ 현재 상태 기반 적응적 분석
        confidence = self._adaptive_gait_verification(vm)
        
        # ④ 상태별 임계값 적용
        current_threshold = self.thresholds['confidence_gait'] if self.current_state == "gait" else self.thresholds['confidence_non_gait']
        
        if confidence > current_threshold:
            return "gait", confidence
        else:
            return "non-gait", 1.0 - confidence
    
    def _adaptive_gait_verification(self, vm: np.ndarray) -> float:
        """적응적 보행 검증 (현재 상태 고려)"""
        confidence = 0.0
        
        # 기본 활동 검사 (더 엄격한 기준)
        I_act = (vm > self.global_thr).astype(int)
        activity_ratio = I_act.mean()
        
        if activity_ratio < 0.45:  # 활동이 부족하면 바로 non-gait (중간 수준)
            return 0.1
        
        # 조건 1: 피크 강도 검사 (상태별 다른 기준, 중간 수준)
        peak_ratio = (vm > self.peak_thr).mean()
        if self.current_state == "gait":
            # gait 상태: 적당히 관대한 기준
            if peak_ratio <= 0.065:  # 6.5%까지 허용 (5%와 8%의 중간)
                confidence += 0.4
        else:
            # non-gait 상태: 중간 기준
            if peak_ratio <= 0.0325:  # 3.25% (2.5%와 4%의 중간)
                confidence += 0.4
        
        # 조건 2: 주파수 분석
        try:
            f, pxx = sg.welch(vm, self.FS, nperseg=min(len(vm), 60))
            
            if len(pxx) > 1:
                peak_idx = np.argmax(pxx[1:]) + 1
                f_peak = f[peak_idx]
                peak_power = pxx[peak_idx]
                total_power = pxx.sum()
                
                # 주파수 범위 (중간 수준)
                freq_range = (0.65, 3.35) if self.current_state == "gait" else (0.85, 2.9)
                if freq_range[0] <= f_peak <= freq_range[1]:
                    confidence += 0.3
                
                # 피크 파워 비율 (중간 수준)
                power_threshold = 0.135 if self.current_state == "gait" else 0.165
                if peak_power >= power_threshold * total_power:
                    confidence += 0.3
                    
        except:
            confidence *= 0.7
        
        return confidence
    
    def _integrated_state_update(self):
        """통합된 상태 업데이트 (Majority Voting + Hysteresis)"""
        # 분석 가능한 최소 프레임 수 확인
        if len(self.frame_labels) < self.hysteresis_frames['start']:
            return
        
        # Majority Voting 기반 의사결정
        decision = self._majority_voting_decision()
        
        # 히스테리시스 적용하여 최종 상태 결정
        new_state = self._apply_hysteresis(decision)
        
        # 상태 변화 시 로깅
        if new_state != self.current_state:
            gait_count = list(self.frame_labels)[-self.decision_window:].count("gait") if len(self.frame_labels) >= self.decision_window else list(self.frame_labels).count("gait")
            total_count = min(len(self.frame_labels), self.decision_window)
            ratio = gait_count / total_count if total_count > 0 else 0
            
            print(f"State changed: {self.current_state} → {new_state}")
            print(f"  Decision basis: {self.debug_info['decision_basis']}")
            print(f"  Gait ratio: {ratio:.2f} ({gait_count}/{total_count})")
            print(f"  Frame: {len(self.frame_labels)}")
            
            self.current_state = new_state
    
    def _majority_voting_decision(self) -> str:
        """적응적 Majority Voting"""
        if len(self.frame_labels) < self.hysteresis_frames['start']:
            return "insufficient_data"
        
        # 현재 상태에 따른 분석 윈도우 결정
        if self.current_state == "non-gait":
            # 보행 시작 감지: 60프레임 윈도우
            window_size = self.hysteresis_frames['start']
            threshold = self.thresholds['gait_start']
        else:
            # 보행 유지/종료 감지: 90프레임 윈도우  
            window_size = self.hysteresis_frames['end']
            # 현재 평균 confidence 기반 dynamic threshold
            recent_confidences = list(self.frame_confidences)[-window_size:]
            avg_confidence = np.mean([c for c in recent_confidences if c > 0])
            
            if avg_confidence > 0.7:
                threshold = self.thresholds['gait_maintain']  # 높은 신뢰도: 관대
            else:
                threshold = (self.thresholds['gait_maintain'] + self.thresholds['gait_start']) / 2  # 낮은 신뢰도: 중간
        
        # 해당 윈도우에서 gait 비율 계산
        recent_labels = list(self.frame_labels)[-window_size:]
        gait_count = recent_labels.count("gait")
        gait_ratio = gait_count / len(recent_labels)
        
        # 디버깅 정보 저장
        self.debug_info = {
            'gait_ratio': gait_ratio,
            'decision_basis': f'window={window_size}, threshold={threshold:.2f}, confidence_avg={avg_confidence:.2f}' if self.current_state == "gait" else f'window={window_size}, threshold={threshold:.2f}',
            'frames_analyzed': len(recent_labels)
        }
        
        # 의사결정
        if gait_ratio >= threshold:
            return "gait"
        elif gait_ratio <= (1.0 - threshold):  # 반대 비율로 non-gait 결정
            return "non-gait"
        else:
            return "uncertain"
    
    def _apply_hysteresis(self, decision: str) -> str:
        """히스테리시스 적용"""
        if decision == "insufficient_data" or decision == "uncertain":
            return self.current_state
        
        # 상태별 전환 조건
        if self.current_state == "non-gait" and decision == "gait":
            # 보행 시작: 빠른 반응 (2초)
            return "gait"
        elif self.current_state == "gait" and decision == "non-gait":
            # 보행 종료: 신중한 반응 (3초 + 엄격한 기준)
            if len(self.frame_labels) >= self.hysteresis_frames['end']:
                recent_labels = list(self.frame_labels)[-self.hysteresis_frames['end']:]
                non_gait_ratio = recent_labels.count("non-gait") / len(recent_labels)
                if non_gait_ratio >= self.thresholds['gait_end']:
                    return "non-gait"
        
        return self.current_state

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
            
            # 추가 디버깅 정보
            gait_ratio = detector.debug_info.get('gait_ratio', 0.0)
            decision_basis = detector.debug_info.get('decision_basis', 'N/A')
            
            print(f"[{state:8s}] {label:8s} ({confidence:.2f}) | "
                  f"a=({accel[0]:+.3f}, {accel[1]:+.3f}, {accel[2]:+.3f}) | "
                  f"f={frame_count:4d} t={frame_time:.2f}s | "
                  f"ratio={gait_ratio:.2f} | {decision_basis}")
            
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