# 라즈베리파이 환경 설정 가이드

## 1. 시스템 요구사항

### 하드웨어
- Raspberry Pi 4 이상 권장 (2GB RAM 이상)
- microSD 카드 (32GB 이상)
- 센서: 6축 IMU (가속도계 + 자이로스코프)

### 소프트웨어
- Raspberry Pi OS (Bullseye 이상)
- Python 3.9 이상

## 2. 환경 설정

### 2.1 시스템 업데이트
```bash
sudo apt update
sudo apt upgrade -y
```

### 2.2 Python 가상환경 설정
```bash
# 프로젝트 디렉토리 이동
cd ~/Gait

# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate
```

### 2.3 패키지 설치

#### 옵션 1: TFLite Runtime만 (권장 - NumPy 2.0 호환)
```bash
# TensorFlow가 설치되어 있다면 먼저 제거
pip uninstall tensorflow -y

# TFLite Runtime만 설치
pip install -r requirements_raspberry_tflite.txt
```

#### 옵션 2: 전체 TensorFlow (NumPy 1.x 필요)
```bash
# NumPy 다운그레이드 필요
pip uninstall numpy -y
pip install -r requirements_raspberry.txt
```

#### 옵션 3: 최소 패키지 (메모리 제한 시)
```bash
pip install -r requirements_minimal.txt
```

## 3. 테스트 실행

### 3.1 시스템 테스트
```bash
python test_gait_system.py
```

### 3.2 예상 결과
```
🎉 All tests passed successfully!
   Ready to use real-time gait detection on Raspberry Pi.
```

## 4. 문제 해결

### 4.1 NumPy 2.0 호환성 문제
**증상**: `AttributeError: _ARRAY_API not found` 또는 NumPy 호환성 경고

**해결방법 1** (권장 - TFLite Runtime만 사용):
```bash
# TensorFlow 완전 제거
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y

# TFLite Runtime만 설치
pip install -r requirements_raspberry_tflite.txt
```

**해결방법 2** (NumPy 다운그레이드):
```bash
# NumPy 다운그레이드
pip uninstall numpy -y
pip install "numpy>=1.24.4,<2.0.0"
pip install -r requirements_raspberry.txt
```

### 4.2 TensorFlow/TFLite 설치 오류
```bash
# TensorFlow 설치 실패 시 TFLite Runtime만 설치
pip uninstall tensorflow
pip install tflite-runtime
```

### 4.2 메모리 부족 오류
```bash
# 스왑 메모리 증가
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048로 변경
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 4.3 scikit-learn 버전 호환성 경고
- 경고가 나타나도 정상 작동합니다
- 스케일러 재생성이 필요한 경우 preprocessing.py 실행

## 5. 성능 최적화

### 5.1 CPU 성능 모드
```bash
# 성능 모드로 설정
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 5.2 메모리 분할 조정
```bash
# GPU 메모리 최소화 (config.txt에 추가)
echo "gpu_mem=16" | sudo tee -a /boot/config.txt
```

## 6. 배포 준비 완료 확인

✅ 모든 테스트 통과  
✅ 실시간 추론 속도 확인  
✅ 메모리 사용량 모니터링  
✅ 센서 연결 테스트  

## 7. 다음 단계

1. 실제 센서 연결 및 데이터 수집
2. 실시간 보행 감지 시스템 구현
3. 웹 인터페이스 또는 모바일 앱 연동
4. 데이터 로깅 및 분석 기능 추가 