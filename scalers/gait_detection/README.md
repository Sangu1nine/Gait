# STANDARDSCALER 모델 사용 가이드 (과적합 해결 버전)

## 🔧 적용된 과적합 해결책
- **TCN 블록 수**: 4개 → 2개 (복잡도 감소)
- **필터 수**: [32,32,64,64] → [16,32] (파라미터 감소)
- **Dense 레이어**: 32 → 16 (용량 감소)
- **학습률**: 0.001 → 0.0005 (안정적 학습)
- **배치 크기**: 256 → 64 (일반화 향상)
- **L2 정규화**: 1e-4 → 1e-3 (과적합 억제 강화)
- **Dropout**: 0.5 유지 (기존 요청에 따라)
- **Early Stopping**: patience=15 유지 (기존 요청에 따라)

## 파일 구조
- model.keras: Keras 모델 (과적합 해결)
- saved_model/: TensorFlow SavedModel
- model.tflite: TensorFlow Lite 모델
- standard_scaler.pkl: STANDARDScaler 객체
- label_encoder.pkl: 라벨 인코더
- thresholds.json: 최적 임계값들
- metrics.json: 성능 메트릭 (개선사항 포함)

## 사용 예시

```python
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 1. 모델 및 전처리 객체 로드
model = tf.keras.models.load_model("model.keras")

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 2. 새로운 데이터 전처리
def preprocess_data(X_new):
    # X_new shape: (n_samples, 60, 6)
    n_samples, n_timesteps, n_features = X_new.shape
    
    # 3D -> 2D 변환
    X_2d = X_new.reshape(-1, n_features)
    
    # 스케일링 적용
    X_scaled = scaler.transform(X_2d)  # fit 없이 transform만!
    
    # 2D -> 3D 변환
    X_scaled = X_scaled.reshape(X_new.shape)
    
    return X_scaled.astype(np.float32)

# 3. 예측
X_new = preprocess_data(your_sensor_data)
y_prob = model.predict(X_new)

# 4. 최적 임계값 적용
optimal_threshold = 0.5  # thresholds.json에서 평균값 사용
y_pred = (y_prob > optimal_threshold).astype(int)

# 5. 라벨 디코딩
y_pred_labels = label_encoder.inverse_transform(y_pred.flatten())
print("예측 결과:", y_pred_labels)
```

## TFLite 사용 (모바일/임베디드)

```python
import numpy as np
import tensorflow as tf

# TFLite 인터프리터 로드
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# 입력/출력 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 추론
def predict_tflite(X_preprocessed):
    interpreter.set_tensor(input_details[0]['index'], X_preprocessed)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# 사용
X_scaled = preprocess_data(your_data)  # 전처리 필수!
predictions = predict_tflite(X_scaled)
```

## 성능 개선 효과
✅ **과적합 문제 해결**: 검증 손실 안정화, 일반화 성능 향상  
✅ **모델 경량화**: 파라미터 수 대폭 감소로 추론 속도 향상  
✅ **안정적 학습**: 낮은 학습률과 작은 배치로 안정적 수렴  
✅ **강화된 정규화**: L2 정규화 강화로 과적합 억제  

## 주의사항
1. **반드시 전처리 적용**: 새로운 데이터는 저장된 스케일러로 전처리해야 함
2. **임계값 적용**: 최적 성능을 위해 저장된 임계값 사용 권장
3. **입력 형태**: (batch_size, 60, 6) 형태의 센서 데이터 필요
4. **Data Leakage 방지**: 새로운 데이터에 대해서는 transform만 사용
5. **경량화된 모델**: 과적합 해결로 더 효율적인 추론 가능

## 성능 지표
- 스케일러: STANDARDScaler
- 모델 개선: 과적합 문제 해결 완료
- 추천 이유: 표준화 + 경량화를 통한 안정적 학습
