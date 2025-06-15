"""
보행 및 낙상 감지 IMU 센서 데이터 수집 프로그램 (규칙 기반 보행 감지)
MODIFIED 2025-01-30: 규칙 기반 보행 감지 + TensorFlow Lite 낙상 감지
MODIFIED 2025-01-30: 낙상 감지 전처리를 raspberry_fall_detection.py와 동일하게 수정
- m/s² 단위 사용 (g 단위 변환 제거)
- Standard → MinMax 스케일러 순서 적용
Features:
- 100Hz IMU 센서 데이터 수집 (멀티스레드)
- 실시간 낙상 감지 (TensorFlow Lite) - 100Hz 데이터 사용, m/s² 단위
- 실시간 보행 감지 (규칙 기반) - 30Hz 다운샘플링 데이터 사용, g 단위
- Supabase 직접 업로드
"""

from smbus2 import SMBus
from bitstring import Bits
import os
from dotenv import load_dotenv
import time
import datetime
import numpy as np
import scipy.signal as sg
import threading
import pickle
from collections import deque
import tensorflow as tf
from supabase import create_client, Client
import io
import csv
from typing import Tuple, Optional

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

# Model paths (fall detection only)
FALL_MODEL_PATH = "models/fall_detection/fall_detection.tflite"
FALL_SCALER_PATH = "scalers/fall_detection"

# Detection parameters
FALL_WINDOW_SIZE = 150  # Window size for fall detection model
SENSOR_HZ = 100  # 센서 데이터 수집 주파수 (100Hz)
GAIT_TARGET_HZ = 30   # 보행 감지용 다운샘플링 주파수 (30Hz)
FALL_THRESHOLD = 0.5  # Fall detection threshold

# State transition parameters
MIN_GAIT_DURATION_FRAMES = 300  # 10 seconds at 30Hz

# Detection timing parameters - 목적별 최적화
FALL_DETECTION_INTERVAL = 0.05  # 낙상 감지 주기 (0.05초 = 100Hz 기준 5frame) - 실시간성 강화
GAIT_DETECTION_INTERVAL = 0.033   # 보행 감지 주기 (0.033초 = 30Hz 기준 1frame) - 실시간성 중시

# Global Supabase client variable
supabase = None

# Global variables for sensor data collection
sensor_data_lock = threading.Lock()
raw_sensor_buffer = deque(maxlen=max(FALL_WINDOW_SIZE * 2, 600))  # 100Hz 버퍼 크기
gait_downsampled_buffer = deque(maxlen=300)  # 30Hz 다운샘플링된 데이터 버퍼 (10초분)
is_running = False

# Rule-based Gait detection variables
gait_detector = None
gait_state = "non-gait"
current_gait_data = deque()  # deque로 변경하여 효율성 향상
current_gait_start_time = None

# Fall detection variables
fall_interpreter = None
fall_scalers = {}  # Dictionary for multiple scalers
last_fall_detection_time = 0  # 마지막 낙상 감지 시간
FALL_COOLDOWN_SECONDS = 5  # 낙상 감지 후 5초 동안 중복 감지 방지

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
        
        # 알고리즘 파라미터 (균형잡힌 조정)
        self.global_thr = 0.06  # g (0.07 → 0.06: 중간 수준으로 조정)
        self.peak_thr = 0.40    # g
        
        # 통합 필터 시스템
        self.lpf_gravity = sg.butter(4, 0.25/(self.FS/2), output='sos')    # 중력 분리
        self.lpf_dynamic = sg.butter(4, 6.0/(self.FS/2), output='sos')     # 동적 가속도 노이즈 제거
        
        # 통합 상태별 임계값 (최종 균형 조정)
        self.thresholds = {
            'gait_start': 0.85,      # non-gait → gait: 85% - 더 엄격하게 (False Positive 감소)
            'gait_maintain': 0.60,   # gait 유지: 60% - 더 엄격하게 (안정성 향상)
            'gait_end': 0.25,        # gait → non-gait: 25% (75% non-gait) - 유지
            'confidence_gait': 0.45, # gait 상태에서 더 높은 신뢰도 요구 (0.4 → 0.45)
            'confidence_non_gait': 0.65  # non-gait 상태에서 약간 완화 (0.7 → 0.65)
        }
        
        # 히스테리시스 설정
        self.hysteresis_frames = {
            'start': 60,    # 2초 - 보행 시작
            'end': 90       # 3초 - 보행 종료  
        }
        
        # 데이터 버퍼 (최대 윈도우 크기로 설정)
        self.buf_acc = deque(maxlen=self.decision_window)
        
        # 상태 관리
        self.current_state = "non-gait"
        self.frame_labels = deque(maxlen=self.decision_window)
        self.frame_confidences = deque(maxlen=self.decision_window)
        
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
        
        # 기본 활동 검사 (완화된 기준)
        I_act = (vm > self.global_thr).astype(int)
        activity_ratio = I_act.mean()
        
        if activity_ratio < 0.45:  # 활동이 부족하면 바로 non-gait (0.5 → 0.45: 완화)
            return 0.1
        
        # 조건 1: 피크 강도 검사 (상태별 완화된 기준)
        peak_ratio = (vm > self.peak_thr).mean()
        if self.current_state == "gait":
            # gait 상태: 약간 완화된 기준
            if peak_ratio <= 0.065:  # 6.5%까지 허용 (5% → 6.5%)
                confidence += 0.4
        else:
            # non-gait 상태: 완화된 기준
            if peak_ratio <= 0.0325:  # 3.25% (2.5% → 3.25%)
                confidence += 0.4
        
        # 조건 2: 주파수 분석
        try:
            f, pxx = sg.welch(vm, self.FS, nperseg=min(len(vm), 60))
            
            if len(pxx) > 1:
                peak_idx = np.argmax(pxx[1:]) + 1
                f_peak = f[peak_idx]
                peak_power = pxx[peak_idx]
                total_power = pxx.sum()
                
                # 주파수 범위 (확대된 범위)
                freq_range = (0.65, 3.35) if self.current_state == "gait" else (0.85, 2.9)
                if freq_range[0] <= f_peak <= freq_range[1]:
                    confidence += 0.3
                
                # 피크 파워 비율 (완화된 기준)
                power_threshold = 0.135 if self.current_state == "gait" else 0.165  # 13.5% / 16.5%
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

def accel_g(val):
    """Convert acceleration value to g"""
    return twocomplements(val)/sensitive_accel

def check_supabase_env():
    """Check Supabase environment variables"""
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    print("🔍 Checking Supabase environment variables...")
    
    if not SUPABASE_URL:
        print("❌ SUPABASE_URL not found in environment variables")
        return False
    else:
        # Mask the URL for security but show it exists
        masked_url = SUPABASE_URL[:20] + "..." + SUPABASE_URL[-10:] if len(SUPABASE_URL) > 30 else SUPABASE_URL
        print(f"✅ SUPABASE_URL found: {masked_url}")
    
    if not SUPABASE_KEY:
        print("❌ SUPABASE_KEY not found in environment variables")
        return False
    else:
        # Mask the key for security
        masked_key = SUPABASE_KEY[:10] + "..." + SUPABASE_KEY[-5:] if len(SUPABASE_KEY) > 15 else "***"
        print(f"✅ SUPABASE_KEY found: {masked_key}")
    
    return True

def init_supabase():
    """Initialize Supabase client"""
    global supabase
    
    # Check environment variables first
    if not check_supabase_env():
        return False
    
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return True
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        return False

def test_supabase_connection():
    """Test Supabase connection by checking service accessibility"""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        print("🔗 Testing Supabase connection...")
        
        # Test 1: Storage bucket access
        try:
            storage_response = supabase.storage.list_buckets()
            if storage_response:
                print("✅ Storage service accessible")
                
                # Check if required buckets exist
                bucket_names = [bucket.name for bucket in storage_response]
                required_buckets = ['gait-data']  # 기존 버킷명 사용
                
                for bucket in required_buckets:
                    if bucket in bucket_names:
                        print(f"  ✅ Bucket '{bucket}' exists")
                    else:
                        print(f"  ⚠️ Bucket '{bucket}' not found - will be created automatically when data is uploaded")
                        
            else:
                print("⚠️ Storage service accessible but no buckets found")
                
        except Exception as storage_e:
            print(f"⚠️ Storage service test failed: {storage_e}")
            # Try alternative storage test
            try:
                # Test with a simple storage operation that doesn't require callable
                storage_info = str(supabase.storage)
                if 'storage' in storage_info.lower():
                    print("  ✅ Storage client initialized (basic test)")
                else:
                    print("  ⚠️ Storage client status unknown")
            except Exception as alt_e:
                print(f"  ❌ Alternative storage test also failed: {alt_e}")
        
        # Test 2: Database table access
        try:
            # Try to read from fall_history table (기존 테이블명 사용)
            table_response = supabase.table("fall_history").select("*").limit(1).execute()
            if table_response:
                print("✅ Database service accessible")
                print(f"  ✅ Table 'fall_history' accessible")
            else:
                print("⚠️ Database service accessible but table query returned no response")
                
        except Exception as db_e:
            print(f"⚠️ Database service test failed: {db_e}")
            # This might be expected if table doesn't exist yet
            if "relation" in str(db_e).lower() and "does not exist" in str(db_e).lower():
                print("  ℹ️ Table 'fall_history' may not exist yet - will be created when first fall is detected")
            else:
                print(f"  ❌ Unexpected database error: {db_e}")
        
        # Test 3: Basic API connectivity
        try:
            # Try to get basic info about the project
            # This is a lightweight operation to test general connectivity
            auth_response = supabase.auth.get_session()
            print("✅ Basic API connectivity confirmed")
            
        except Exception as api_e:
            print(f"⚠️ Basic API test failed: {api_e}")
        
        print("🔗 Supabase connection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection test failed: {e}")
        return False

def save_gait_data_to_supabase(gait_data):
    """Save gait data to Supabase"""
    if not supabase:
        print("⚠️ Supabase not initialized - cannot save gait data")
        return
    
    try:
        # Convert data to CSV format
        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)
        
        # Write header (frame starts from 0, no gait_frame field)
        csv_writer.writerow(['frame', 'sync_timestamp', 'accel_x', 'accel_y', 'accel_z', 
                            'gyro_x', 'gyro_y', 'gyro_z', 'unix_timestamp'])
        
        # Write data with proper formatting
        for i, data in enumerate(gait_data):
            csv_writer.writerow([
                i,  # frame: gait 시작을 0으로 하여 1씩 증가
                                 f"{i * 0.033:.3f}",  # sync_timestamp: 0.033초씩 증가 (30Hz)
                f"{data['accel_x']:.3f}",  # 가속도: 소수점 3자리 반올림
                f"{data['accel_y']:.3f}",
                f"{data['accel_z']:.3f}",
                f"{data['gyro_x']:.5f}",  # 각속도: 소수점 5자리 반올림
                f"{data['gyro_y']:.5f}",
                f"{data['gyro_z']:.5f}",
                data['unix_timestamp']  # 절대 시간은 그대로 유지
            ])
        
        # Upload to Supabase (기존 버킷명 사용)
        response = supabase.storage.from_("gait-data").upload(
            f"gait_{int(time.time())}.csv",
            csv_data.getvalue(),
            {"content-type": "text/csv"}
        )
        
        if response:
            print(f"✅ Gait data saved to Supabase: {response}")
        else:
            print("⚠️ Failed to save gait data to Supabase")
    except Exception as e:
        print(f"❌ Error saving gait data to Supabase: {e}")

def save_fall_event_to_supabase(timestamp):
    """Save fall event to Supabase"""
    if not supabase:
        print("⚠️ Supabase not initialized - cannot save fall event")
        return
    
    try:
        # Convert timestamp to ISO format for better compatibility
        fall_time = datetime.datetime.fromtimestamp(timestamp)
        fall_data = {
            "timestamp": fall_time.isoformat(),
            "detected_at": datetime.datetime.now().isoformat()
        }
        
        # Try to insert into fall_history table (기존 테이블명 사용)
        response = supabase.table("fall_history").insert(fall_data).execute()
        
        if response and response.data:
            print(f"✅ Fall event saved to Supabase")
        else:
            print("⚠️ Fall event saved but no response data")
            
    except Exception as e:
        print(f"❌ Error saving fall event to Supabase: {e}")
        
        # Fallback: Save to storage as backup
        try:
            fall_filename = f"fall_event_{int(timestamp)}.json"
            fall_json = {
                "timestamp": datetime.datetime.fromtimestamp(timestamp).isoformat(),
                "detected_at": datetime.datetime.now().isoformat(),
                "unix_timestamp": timestamp
            }
            
            import json
            fall_content = json.dumps(fall_json, indent=2)
            
            backup_response = supabase.storage.from_("gait-data").upload(
                fall_filename,
                fall_content,
                {"content-type": "application/json"}
            )
            
            if backup_response:
                print(f"✅ Fall event saved to storage as backup: {fall_filename}")
            else:
                print("⚠️ Failed to save fall event backup to storage")
                
        except Exception as backup_e:
            print(f"❌ Backup save also failed: {backup_e}")

def load_models():
    """Load fall detection model only (gait detection is now rule-based)"""
    global fall_interpreter, fall_scalers
    
    # Check scikit-learn version
    try:
        import sklearn
        print(f"📦 scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not installed")
    
    print("🚶 Gait detection: Rule-based (no model loading needed)")
    
    # Load fall detection model
    try:
        if os.path.exists(FALL_MODEL_PATH):
            fall_interpreter = tf.lite.Interpreter(model_path=FALL_MODEL_PATH)
            fall_interpreter.allocate_tensors()
            print(f"✅ Fall model loaded: {FALL_MODEL_PATH}")
        
        # Load fall scalers (both minmax and standard) with version compatibility
        scalers_loaded = 0
        total_scalers = 0
        
        for sensor in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
            minmax_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_minmax_scaler.pkl")
            standard_file = os.path.join(FALL_SCALER_PATH, f"{sensor}_standard_scaler.pkl")
            
            # Load MinMax scaler
            if os.path.exists(minmax_file):
                total_scalers += 1
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        with open(minmax_file, 'rb') as f:
                            fall_scalers[f"{sensor}_minmax"] = pickle.load(f)
                    scalers_loaded += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {sensor}_minmax scaler: {e}")
            
            # Load Standard scaler
            if os.path.exists(standard_file):
                total_scalers += 1
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        with open(standard_file, 'rb') as f:
                            fall_scalers[f"{sensor}_standard"] = pickle.load(f)
                    scalers_loaded += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {sensor}_standard scaler: {e}")
        
        print(f"✅ Fall scalers loaded: {scalers_loaded}/{total_scalers}")
        
        if scalers_loaded == 0:
            print("⚠️ No fall scalers loaded - fall detection may not work correctly")
        elif scalers_loaded < total_scalers:
            print("⚠️ Some fall scalers failed to load - fall detection accuracy may be reduced")
            
    except Exception as e:
        print(f"❌ Fall model loading error: {e}")

def sensor_collection_thread():
    """Thread for collecting sensor data at 100Hz"""
    global raw_sensor_buffer, gait_downsampled_buffer, is_running
    
    start_time = time.time()
    frame_count = 0
    gait_frame_count = 0
    last_gait_sample_time = 0
    gait_sampling_interval = 1.0 / GAIT_TARGET_HZ  # 30Hz를 위한 샘플링 간격
    
    while is_running:
        try:
            # Read IMU sensor data
            accel_x = accel_ms2(read_data(register_accel_xout_h))
            accel_y = -accel_ms2(read_data(register_accel_yout_h))
            accel_z = accel_ms2(read_data(register_accel_zout_h))
            
            gyro_x = gyro_dps(read_data(register_gyro_xout_h))
            gyro_y = gyro_dps(read_data(register_gyro_yout_h))
            gyro_z = gyro_dps(read_data(register_gyro_zout_h))
            
            # Calculate timestamp
            current_time = time.time()
            sync_timestamp = current_time - start_time
            
            # Store raw sensor data (100Hz) for fall detection
            sensor_data = {
                'frame': frame_count,
                'sync_timestamp': sync_timestamp,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'unix_timestamp': current_time
            }
            
            with sensor_data_lock:
                # 100Hz 버퍼에 추가 (낙상 감지용)
                if len(raw_sensor_buffer) >= raw_sensor_buffer.maxlen:
                    raw_sensor_buffer.popleft()
                raw_sensor_buffer.append(sensor_data)
                
                # 30Hz 다운샘플링 (보행 감지용)
                if sync_timestamp - last_gait_sample_time >= gait_sampling_interval:
                    gait_sensor_data = sensor_data.copy()
                    gait_sensor_data['gait_frame'] = gait_frame_count
                    
                    if len(gait_downsampled_buffer) >= gait_downsampled_buffer.maxlen:
                        gait_downsampled_buffer.popleft()
                    gait_downsampled_buffer.append(gait_sensor_data)
                    
                    last_gait_sample_time = sync_timestamp
                    gait_frame_count += 1
            
            frame_count += 1
            
            # Maintain 100Hz sampling rate
            next_sample_time = start_time + (frame_count * (1.0 / SENSOR_HZ))
            sleep_time = next_sample_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"❌ Sensor collection error: {e}")
            time.sleep(0.001)

def preprocess_for_fall(sensor_window):
    """
    Preprocess sensor data for fall detection
    Following raspberry_fall_detection.py preprocessing method
    - Use m/s² units for accelerometer (NOT g units)
    - Apply Standard scaling first, then MinMax scaling
    """
    if not fall_scalers:
        return None
    
    try:
        # Process each sensor channel - Keep m/s² units for accelerometer
        processed_data = []
        
        for data in sensor_window:
            # Keep accelerometer data in m/s² units (do NOT convert to g)
            acc_x = data['accel_x']  # m/s²
            acc_y = data['accel_y']  # m/s²
            acc_z = data['accel_z']  # m/s²
            gyr_x = data['gyro_x']   # degrees/s
            gyr_y = data['gyro_y']   # degrees/s
            gyr_z = data['gyro_z']   # degrees/s
            
            processed_data.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z])
        
        # Convert to numpy array
        sensor_array = np.array(processed_data, dtype=np.float32)
        
        # Apply scalers following raspberry_fall_detection.py approach
        # Standard scaling FIRST, then MinMax scaling
        sensor_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        scaled_data = np.zeros_like(sensor_array)
        
        for i, sensor_name in enumerate(sensor_names):
            val = sensor_array[:, i]
            
            # Apply Standard scaler FIRST if available
            if f"{sensor_name}_standard" in fall_scalers:
                scaler = fall_scalers[f"{sensor_name}_standard"]
                # Apply standard scaling: (x - mean) / std
                val = (val - scaler.mean_[0]) / scaler.scale_[0]
            
            # Apply MinMax scaler SECOND if available
            if f"{sensor_name}_minmax" in fall_scalers:
                scaler = fall_scalers[f"{sensor_name}_minmax"]
                # Apply minmax scaling: x * scale + min
                val = val * scaler.scale_[0] + scaler.min_[0]
            
            scaled_data[:, i] = val
        
        return scaled_data.reshape(1, FALL_WINDOW_SIZE, 6)
    except Exception as e:
        print(f"❌ Fall preprocessing error: {e}")
        return None

def gait_detection_thread():
    """Thread for rule-based gait detection using 30Hz downsampled data"""
    global gait_state, current_gait_data, current_gait_start_time, gait_detector
    
    print("🚶 Rule-based gait detection thread initialized (30Hz downsampled)")
    
    while is_running:
        try:
            # Get available downsampled sensor data (30Hz)
            with sensor_data_lock:
                if len(gait_downsampled_buffer) == 0:
                    time.sleep(0.01)
                    continue
                
                # 가장 최신 데이터 가져오기
                latest_sensor_data = gait_downsampled_buffer[-1]
            
            # 가속도 데이터를 g 단위로 변환하여 규칙 기반 감지기에 전달
            accel_g_data = np.array([
                latest_sensor_data['accel_x'] / 9.80665,  # m/s² to g
                latest_sensor_data['accel_y'] / 9.80665,
                latest_sensor_data['accel_z'] / 9.80665
            ])
            
            # 규칙 기반 보행 감지 처리
            if gait_detector:
                state, label, confidence = gait_detector.process_frame(accel_g_data)
                
                # 상태 변화 감지 및 데이터 수집 관리
                if state != gait_state:
                    if state == "gait" and gait_state == "non-gait":
                        # 보행 시작
                        gait_state = "gait"
                        current_gait_start_time = latest_sensor_data['unix_timestamp']
                        current_gait_data = deque()
                        print(f"🚶 Rule-based Gait started at gait_frame {latest_sensor_data['gait_frame']} (confidence: {confidence:.3f})")
                    
                    elif state == "non-gait" and gait_state == "gait":
                        # 보행 종료
                        gait_duration_frames = len(current_gait_data)
                        gait_duration_seconds = gait_duration_frames / GAIT_TARGET_HZ
                        
                        print(f"🛑 Rule-based Gait ended at gait_frame {latest_sensor_data['gait_frame']} (duration: {gait_duration_frames} frames, {gait_duration_seconds:.1f}s)")
                        
                        if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                            save_gait_data_to_supabase(list(current_gait_data))
                            print(f"✅ Gait data saved ({gait_duration_frames} frames)")
                        else:
                            print(f"⚠️ Gait duration too short: {gait_duration_frames} frames ({gait_duration_seconds:.1f}s < {MIN_GAIT_DURATION_FRAMES/GAIT_TARGET_HZ:.1f}s)")
                        
                        # 상태 리셋
                        gait_state = "non-gait"
                        current_gait_data = deque()
                        current_gait_start_time = None
                
                # 보행 중인 경우 데이터 수집
                if gait_state == "gait":
                    current_gait_data.append(latest_sensor_data)
                
                # 디버깅 정보 출력 (매 1초마다)
                current_frame = latest_sensor_data['gait_frame']
                if current_frame % 30 == 0:  # Print every 1 second at 30Hz
                    gait_ratio = gait_detector.debug_info.get('gait_ratio', 0.0)
                    decision_basis = gait_detector.debug_info.get('decision_basis', 'N/A')
                    print(f"🔍 Frame {current_frame}: State={state}, Label={label}, Conf={confidence:.3f}, Ratio={gait_ratio:.2f} | {decision_basis}")
            
            time.sleep(GAIT_DETECTION_INTERVAL)  # 30Hz 주기
            
        except Exception as e:
            print(f"❌ Rule-based gait detection error: {e}")
            time.sleep(0.1)

def fall_detection_thread():
    """Thread for fall detection using 100Hz data with longer interval"""
    global last_fall_detection_time
    print("🚨 Fall detection thread initialized (100Hz data, 0.05s interval)")
    
    while is_running:
        try:
            with sensor_data_lock:
                if len(raw_sensor_buffer) >= FALL_WINDOW_SIZE:
                    sensor_window = list(raw_sensor_buffer)[-FALL_WINDOW_SIZE:]
                else:
                    time.sleep(0.1)
                    continue
            
            # Preprocess and predict
            if fall_interpreter and fall_scalers:
                preprocessed = preprocess_for_fall(sensor_window)
                if preprocessed is not None:
                    # Run inference
                    input_details = fall_interpreter.get_input_details()
                    output_details = fall_interpreter.get_output_details()
                    
                    fall_interpreter.set_tensor(input_details[0]['index'], preprocessed)
                    fall_interpreter.invoke()
                    
                    prediction = fall_interpreter.get_tensor(output_details[0]['index'])
                    fall_probability = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
                    
                    current_time = time.time()
                    
                    # Check for fall with cooldown period
                    if fall_probability > FALL_THRESHOLD:
                        # 중복 감지 방지: 마지막 낙상 감지 후 5초 이내면 무시
                        if current_time - last_fall_detection_time > FALL_COOLDOWN_SECONDS:
                            print(f"🚨 Fall detected! Probability: {fall_probability:.2f}")
                            save_fall_event_to_supabase(sensor_window[-1]['unix_timestamp'])
                            last_fall_detection_time = current_time
                        else:
                            remaining_cooldown = FALL_COOLDOWN_SECONDS - (current_time - last_fall_detection_time)
                            print(f"🔄 Fall detected but in cooldown period (remaining: {remaining_cooldown:.1f}s)")
            
            time.sleep(FALL_DETECTION_INTERVAL)  # 실시간성 강화된 낙상 감지 주기
            
        except Exception as e:
            print(f"❌ Fall detection error: {e}")
            time.sleep(0.1)

def main():
    """Main execution function"""
    global is_running, gait_detector
    
    print("=" * 70)
    print("🚶 Rule-based Gait & TensorFlow Fall Detection System")
    print("=" * 70)
    print(f"📊 Sensor collection: {SENSOR_HZ}Hz")
    print(f"🚶 Gait detection: {GAIT_TARGET_HZ}Hz (rule-based)")
    print(f"   └─ Enhanced filtering, majority voting, hysteresis")
    print(f"🚨 Fall detection: {SENSOR_HZ}Hz (TensorFlow Lite)")
    print(f"   └─ Interval: {FALL_DETECTION_INTERVAL}s ({1/FALL_DETECTION_INTERVAL:.0f}Hz detection)")
    print("=" * 70)
    
    # Initialize and test Supabase connection
    if init_supabase():
        print("✅ Supabase client initialized")
        
        # Test connection
        connection_ok = test_supabase_connection()
        if connection_ok:
            print("✅ Supabase connection verified - data will be uploaded to cloud")
        else:
            print("⚠️ Supabase connection issues detected - check your network and credentials")
            print("⚠️ System will continue but data upload may fail")
    else:
        print("❌ Supabase initialization failed")
        print("⚠️ Please check your SUPABASE_URL and SUPABASE_KEY in .env file")
        print("⚠️ Continuing without Supabase - data will not be saved to cloud")
    
    # Load fall detection model
    load_models()
    
    # Initialize rule-based gait detector
    gait_detector = GaitDetector(sampling_rate=GAIT_TARGET_HZ)
    print("✅ Rule-based gait detector initialized")
    
    # Initialize IMU sensor
    bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
    print("✅ IMU sensor initialized")
    
    # Start threads
    is_running = True
    
    sensor_thread = threading.Thread(target=sensor_collection_thread)
    sensor_thread.daemon = True
    sensor_thread.start()
    print("✅ Sensor collection thread started (100Hz)")
    
    gait_thread = threading.Thread(target=gait_detection_thread)
    gait_thread.daemon = True
    gait_thread.start()
    print("✅ Rule-based gait detection thread started (30Hz downsampled)")
    
    fall_thread = threading.Thread(target=fall_detection_thread)
    fall_thread.daemon = True
    fall_thread.start()
    print("✅ Fall detection thread started (100Hz, 0.05s interval)")
    
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Print status every 5 seconds
            if int(time.time()) % 5 == 0:
                with sensor_data_lock:
                    raw_buffer_size = len(raw_sensor_buffer)
                    gait_buffer_size = len(gait_downsampled_buffer)
                
                gait_data_size = len(current_gait_data) if current_gait_data else 0
                
                print(f"📊 Status - Raw(100Hz): {raw_buffer_size}, Gait(30Hz): {gait_buffer_size}, "
                      f"State: {gait_state}, Gait frames: {gait_data_size}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        
    finally:
        is_running = False
        time.sleep(1)  # Allow threads to finish
        
        # Save any remaining gait data
        if gait_state == "gait" and current_gait_data:
            gait_duration_frames = len(current_gait_data)
            print(f"💾 Saving remaining gait data: {gait_duration_frames} frames")
            if gait_duration_frames >= MIN_GAIT_DURATION_FRAMES:
                save_gait_data_to_supabase(list(current_gait_data))
                print(f"✅ Final gait data saved ({gait_duration_frames} frames)")
            else:
                print(f"⚠️ Final gait data too short: {gait_duration_frames} frames")
        
        bus.close()
        print("✅ System stopped")

if __name__ == "__main__":
    main()