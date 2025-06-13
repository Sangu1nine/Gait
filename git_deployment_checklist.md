# 라즈베리파이 배포용 필수 파일 체크리스트

## 🔥 필수 파일/폴더 (반드시 포함)

### 1. 메인 실행 파일들
```
📁 프로젝트 루트/
├── realtime_gait_detector.py    # 🔑 메인 실행 파일 (필수)
├── test_gait_system.py          # 🔧 시스템 테스트 스크립트 (권장)
└── README_usage_guide.md        # 📖 사용 가이드 (권장)
```

### 2. 스케일러 파일들 (필수 🚨)
```
📁 scalers/
└── 📁 gait/
    ├── minmax_scaler.pkl      # 🔑 MinMax 스케일러 (724B, 필수)
    ├── standard_scaler.pkl    # 🔑 Standard 스케일러 (594B, 필수)  
    └── metadata.json          # 🔑 메타데이터 (842B, 필수)
```

### 3. 모델 파일들 (필수 🚨)
```
📁 models/
└── 📁 gait_detect/
    └── 📁 saved_model/
        └── model.tflite       # 🔑 TFLite 모델 (필수)
```

## ⚠️ 선택적 파일들 (크기가 크므로 제외 가능)

### 1. 대용량 학습 데이터 (제외 권장)
```
❌ scalers/gait/stage1_data_standard.npy     # 29MB - 제외
❌ scalers/gait/stage2_data.npy              # 113MB - 제외  
❌ scalers/gait/stage1_data_minmax.npy       # 29MB - 제외
❌ scalers/gait/stage2_labels.npy            # 28MB - 제외
❌ scalers/gait/stage1_labels.npy            # 333KB - 제외
❌ scalers/gait/metadata.pkl                 # 722B - json으로 대체됨
```

### 2. 학습 결과 파일들 (제외 가능)
```
❌ models/gait_detect/results/               # 학습 시각화 결과 - 제외 가능
❌ models/gait_detect/saved_model/model.keras # Keras 모델 - TFLite만 있으면 됨
❌ models/gait_detect/saved_model/saved_model/ # TensorFlow SavedModel - TFLite만 있으면 됨
```

## 📋 Git 전송용 최소 구조

```
your_project/
├── realtime_gait_detector.py    # 2-3KB
├── test_gait_system.py          # 8-10KB  
├── README_usage_guide.md        # 15-20KB
├── scalers/
│   └── gait/
│       ├── minmax_scaler.pkl    # 724B
│       ├── standard_scaler.pkl  # 594B
│       └── metadata.json        # 842B
└── models/
    └── gait_detect/
        └── saved_model/
            └── model.tflite     # 크기 확인 필요 (보통 1-5MB)
```

## 🚀 Git 배포 명령어

### 1. .gitignore 파일 생성 (대용량 파일 제외)
```bash
# .gitignore 내용
*.npy
scalers/gait/*.npy
scalers/gait/metadata.pkl
models/gait_detect/results/
models/gait_detect/saved_model/model.keras
models/gait_detect/saved_model/saved_model/
__pycache__/
*.pyc
*.log
```

### 2. Git 초기화 및 커밋
```bash
git init
git add .
git commit -m "Add gait detection system for Raspberry Pi"
git remote add origin your_repository_url
git push -u origin main
```

### 3. 라즈베리파이에서 클론
```bash
git clone your_repository_url
cd your_project
python test_gait_system.py  # 시스템 테스트
```

## 🔍 파일 크기 확인 방법

현재 디렉토리에서 각 파일 크기를 확인하세요:
```bash
# Windows PowerShell
Get-ChildItem -Recurse | Select-Object Name, Length | Sort-Object Length -Descending

# Linux/Mac
find . -type f -exec ls -lh {} + | sort -k5 -hr
```

## ⚡ 최적화 팁

### 1. 필수 파일만 선별
- **반드시 필요**: 스케일러 3개 + TFLite 모델 1개 + 실행 스크립트
- **총 용량**: 약 5-10MB (TFLite 모델 크기에 따라)

### 2. Git LFS 사용 (모델 파일이 큰 경우)
```bash
git lfs track "*.tflite"
git add .gitattributes
```

### 3. 압축 전송 (대안)
```bash
# 필수 파일만 압축
tar -czf gait_detection_minimal.tar.gz \
  realtime_gait_detector.py \
  test_gait_system.py \
  README_usage_guide.md \
  scalers/gait/*.pkl \
  scalers/gait/*.json \
  models/gait_detect/saved_model/*.tflite
```

## 🧪 라즈베리파이 배포 후 테스트

1. **시스템 테스트 실행**
```bash
python test_gait_system.py
```

2. **의존성 설치 (필요시)**
```bash
pip install numpy scipy scikit-learn tensorflow
# 또는 TFLite만
pip install tflite-runtime
```

3. **실시간 시스템 실행**
```bash
python realtime_gait_detector.py
```

## ❗ 주의사항

1. **TFLite 모델 파일 크기 확인**: 보통 1-5MB이지만, 너무 크면 Git LFS 사용
2. **스케일러 파일 버전**: 반드시 학습에 사용된 동일한 스케일러 파일 사용
3. **경로 설정**: 코드에서 상대경로 사용하므로 폴더 구조 유지 필요
4. **권한 설정**: 라즈베리파이에서 실행 권한 부여 (`chmod +x *.py`)

## 📊 전송 용량 예상

- **최소 필수 파일**: 약 2-3MB
- **권장 파일 포함**: 약 5-10MB  
- **대용량 학습 데이터 포함**: 200MB+ (권장하지 않음) 