# 라즈베리파이 실시간 보행 감지 시스템 사용 가이드

## 📋 목차
1. [결과물 분석](#결과물-분석)
2. [파일 구조 설명](#파일-구조-설명)
3. [실시간 시스템 구현](#실시간-시스템-구현)
4. [사용법](#사용법)
5. [문제 해결](#문제-해결)

## 🗂️ 결과물 분석

### `/models/gait_detect` 폴더 (학습된 모델)
```
models/gait_detect/
├── results/              # 학습 결과 및 시각화 파일
└── saved_model/          # TensorFlow SavedModel 형식
    ├── model.tflite      # 🔑 TFLite 모델 (라즈베리파이용)
    ├── model.keras       # Keras 모델
    └── saved_model/      # TensorFlow SavedModel
```

### `/scalers/gait` 폴더 (전처리 도구)
```
scalers/gait/
├── minmax_scaler.pkl     # 🔑 MinMax 스케일러 (필수)
├── standard_scaler.pkl   # 🔑 Standard 스케일러 (필수)
├── metadata.json         # 🔑 전처리 메타데이터 (필수)
├── metadata.pkl          # 메타데이터 (pickle 형식)
├── stage1_data_*.npy     # 전처리된 학습 데이터 (참고용)
└── stage2_*.npy          # Stage2 데이터 (참고용)
```

## 📝 파일 구조 설명

### 1. 스케일러 파일들
- **`minmax_scaler.pkl`**: 센서 데이터를 0~1 범위로 정규화하는 스케일러
- **`standard_scaler.pkl`**: 데이터를 평균 0, 표준편차 1로 표준화하는 스케일러
- **적용 순서**: MinMax → Standard (preprocessing.py와 동일)

### 2. 모델 파일
- **`model.tflite`**: 라즈베리파이용 최적화된 TensorFlow Lite 모델
- **입력 형태**: `(1, 60, 6)` - 1배치, 60프레임, 6축 센서
- **출력**: 보행 확률 (0~1, 0.5 이상이면 보행)

### 3. 처리 파라미터
```python
SAMPLING_RATE = 30      # 30Hz 샘플링
WINDOW_SIZE = 60        # 60프레임 윈도우 (2초)
STRIDE = 30             # 30프레임 간격 (1초 오버랩)
SENSOR_FEATURES = 6     # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
FILTER_CUTOFF = 10      # 10Hz 버터워스 로우패스 필터
```

## 🔧 실시간 시스템 구현

### 1. 필요 라이브러리 설치
```bash
# 라즈베리파이에서 실행
pip install numpy scipy scikit-learn tensorflow

# 센서 연결용 (선택사항)
pip install RPi.GPIO adafruit-circuitpython-lsm6ds
```

### 2. 파일 구조 설정
```
your_project/
├── scalers/
│   └── gait/
│       ├── minmax_scaler.pkl
│       ├── standard_scaler.pkl
│       └── metadata.json
├── models/
│   └── gait_detect/
│       └── saved_model/
│           └── model.tflite
└── realtime_gait_detector.py
```

### 3. 핵심 구현 코드

#### 스케일러 로더
```python
import pickle
import numpy as np

class ScalerProcessor:
    def __init__(self):
        # 스케일러 로드
        with open('scalers/gait/minmax_scaler.pkl', 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        with open('scalers/gait/standard_scaler.pkl', 'rb') as f:
            self.standard_scaler = pickle.load(f)
    
    def transform(self, data):
        """센서 데이터 스케일링 (preprocessing.py와 동일)"""
        # 형태: (60, 6) → (60*6,) → 스케일링 → (60, 6)
        original_shape = data.shape
        data_flat = data.reshape(-1, original_shape[-1])
        
        # MinMax → Standard 순서 적용
        data_minmax = self.minmax_scaler.transform(data_flat)
        data_scaled = self.standard_scaler.transform(data_minmax)
        
        return data_scaled.reshape(original_shape)
```

#### 필터링 모듈
```python
from scipy import signal

class SignalProcessor:
    def __init__(self):
        # 버터워스 필터 설정 (preprocessing.py와 동일)
        fs = 30  # 샘플링 주파수
        cutoff = 10  # 컷오프 주파수
        order = 4  # 필터 차수
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low')
    
    def filter_data(self, data):
        """6축 센서 데이터 필터링"""
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):  # 각 축별로 필터 적용
            filtered[:, i] = signal.filtfilt(self.b, self.a, data[:, i])
        return filtered
```

#### TFLite 추론 모듈
```python
import tensorflow as tf

class GaitPredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, data):
        """보행 예측"""
        # 배치 차원 추가: (60, 6) → (1, 60, 6)
        input_data = np.expand_dims(data, axis=0).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return float(output[0][0])  # 보행 확률 반환
```

## 🚀 사용법

### 1. 시스템 검증
```python
python realtime_gait_detector.py
```
- 스케일러와 모델 파일이 올바르게 로드되는지 확인
- 테스트 데이터로 전체 파이프라인 검증

### 2. 실시간 데이터 처리 흐름
```
센서 데이터 수집 (30Hz)
↓
버터워스 필터링 (10Hz 로우패스)
↓
60프레임 윈도우 생성
↓
스케일링 (MinMax → Standard)
↓
TFLite 모델 추론
↓
보행/비보행 판단 (임계값: 0.5)
```

### 3. 센서 데이터 형식
```python
# 매 프레임마다 다음 형식의 데이터 필요
sensor_data = [
    accel_x,    # 가속도 X축 (m/s²)
    accel_y,    # 가속도 Y축 (m/s²)  
    accel_z,    # 가속도 Z축 (m/s²)
    gyro_x,     # 각속도 X축 (deg/s)
    gyro_y,     # 각속도 Y축 (deg/s)
    gyro_z      # 각속도 Z축 (deg/s)
]
```

### 4. 실제 센서 연결 예제 (LSM6DS 센서 사용)
```python
import board
import adafruit_lsm6ds.lsm6ds33 as lsm6ds33

# I2C 연결
i2c = board.I2C()
sensor = lsm6ds33.LSM6DS33(i2c)

# 데이터 수집 루프
while True:
    # 센서 데이터 읽기
    accel_x, accel_y, accel_z = sensor.acceleration
    gyro_x, gyro_y, gyro_z = sensor.gyro
    
    sensor_data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    # 보행 감지 처리
    result = detector.process_sensor_data(sensor_data)
    if result:
        print(f"보행 감지: {result['is_gait']}, 확률: {result['probability']:.3f}")
    
    time.sleep(1/30)  # 30Hz
```

## 🔧 문제 해결

### 1. 파일 경로 오류
```
❌ FileNotFoundError: 스케일러 파일을 찾을 수 없습니다
```
**해결법**: 파일 경로 확인 및 수정
```python
# realtime_gait_detector.py에서 경로 수정
SCALERS_DIR = "your_actual_path/scalers/gait"
MODELS_DIR = "your_actual_path/models/gait_detect/saved_model"
```

### 2. TensorFlow Lite 설치 오류
```bash
# 라즈베리파이용 TFLite 설치
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### 3. 성능 최적화
- **메모리 최적화**: 윈도우 버퍼 크기 조정
- **CPU 최적화**: 필터링 연산 최적화
- **실시간 처리**: 멀티스레딩 적용

### 4. 정확도 문제
- **센서 캘리브레이션**: 센서 오프셋 보정
- **필터 파라미터**: 노이즈 환경에 따른 필터 조정
- **임계값 조정**: 환경에 맞는 임계값 설정

## 📊 성능 지표

### 전처리 파라미터 (동일하게 적용)
- **샘플링 주파수**: 30Hz
- **윈도우 크기**: 60프레임 (2초)
- **필터**: 4차 버터워스, 10Hz 컷오프
- **스케일링**: MinMax → Standard

### 예상 성능
- **지연시간**: < 50ms (라즈베리파이 4 기준)
- **메모리 사용량**: < 500MB  
- **정확도**: 학습 모델 성능에 따라 결정

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 파일 경로가 올바른지 확인
2. 전처리 파라미터가 학습 시와 동일한지 확인
3. 센서 데이터 형식과 단위가 올바른지 확인
4. TensorFlow Lite 모델이 올바르게 로드되는지 확인 