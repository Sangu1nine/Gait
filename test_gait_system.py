#!/usr/bin/env python3
"""
보행 감지 시스템 테스트 스크립트

전처리된 스케일러와 TFLite 모델이 올바르게 작동하는지 테스트합니다.
"""

import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime

def test_file_existence():
    """필수 파일 존재 확인"""
    print("🔍 파일 존재 확인 중...")
    
    required_files = {
        "MinMax Scaler": "scalers/gait/minmax_scaler.pkl",
        "Standard Scaler": "scalers/gait/standard_scaler.pkl", 
        "Metadata": "scalers/gait/metadata.json",
        "TFLite Model": "models/gait_detect/results/gait_detection.tflite"
    }
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"✅ {name}: {path} ({file_size:,} bytes)")
        else:
            print(f"❌ {name}: {path} (파일 없음)")
            all_exist = False
    
    return all_exist

def test_scalers():
    """스케일러 로드 및 테스트"""
    print("\n🔧 스케일러 테스트 중...")
    
    try:
        # 스케일러 로드
        with open("scalers/gait/minmax_scaler.pkl", 'rb') as f:
            minmax_scaler = pickle.load(f)
        print("✅ MinMaxScaler 로드 성공")
        
        with open("scalers/gait/standard_scaler.pkl", 'rb') as f:
            standard_scaler = pickle.load(f)
        print("✅ StandardScaler 로드 성공")
        
        # 테스트 데이터 생성 (60프레임, 6축 센서)
        test_data = np.random.randn(60, 6) * 10  # 랜덤 센서 데이터
        print(f"📊 테스트 데이터 생성: {test_data.shape}")
        print(f"   원본 범위: {test_data.min():.3f} ~ {test_data.max():.3f}")
        
        # 스케일링 적용 (preprocessing.py와 동일한 순서)
        data_reshaped = test_data.reshape(-1, 6)
        
        # MinMax 스케일링
        data_minmax = minmax_scaler.transform(data_reshaped)
        data_minmax = data_minmax.reshape(60, 6)
        print(f"   MinMax 후: {data_minmax.min():.3f} ~ {data_minmax.max():.3f}")
        
        # Standard 스케일링
        data_minmax_reshaped = data_minmax.reshape(-1, 6)
        data_scaled = standard_scaler.transform(data_minmax_reshaped)
        data_scaled = data_scaled.reshape(60, 6)
        print(f"   Standard 후: {data_scaled.min():.3f} ~ {data_scaled.max():.3f}")
        
        print("✅ 스케일러 테스트 성공")
        return data_scaled
        
    except Exception as e:
        print(f"❌ 스케일러 테스트 실패: {e}")
        return None

def test_metadata():
    """메타데이터 로드 및 확인"""
    print("\n📋 메타데이터 확인 중...")
    
    try:
        # JSON 형식 시도
        if os.path.exists("scalers/gait/metadata.json"):
            with open("scalers/gait/metadata.json", 'r') as f:
                metadata = json.load(f)
        else:
            # PKL 형식 시도
            with open("scalers/gait/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
        
        print("✅ 메타데이터 로드 성공")
        print(f"   타임스탬프: {metadata.get('timestamp', 'N/A')}")
        print(f"   Stage1 데이터 형태: {metadata.get('stage1_shape', 'N/A')}")
        print(f"   Walking 피험자 수: {metadata.get('n_walking_subjects', 'N/A')}")
        print(f"   Non-walking 피험자 수: {metadata.get('n_non_walking_subjects', 'N/A')}")
        
        if 'stage1_label_distribution' in metadata:
            label_dist = metadata['stage1_label_distribution']
            print(f"   라벨 분포: {label_dist}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ 메타데이터 로드 실패: {e}")
        return None

def test_tflite_model(test_data):
    """TFLite 모델 테스트"""
    print("\n🤖 TFLite 모델 테스트 중...")
    
    try:
        # TensorFlow Lite 가져오기 시도
        try:
            import tensorflow as tf
            print("✅ TensorFlow 사용")
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                tf = tflite
                print("✅ TFLite Runtime 사용")
            except ImportError:
                print("❌ TensorFlow 또는 TFLite Runtime이 필요합니다")
                return False
        
        # 모델 로드
        model_path = "models/gait_detect/results/gait_detection.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 입출력 정보 확인
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TFLite 모델 로드 성공")
        print(f"   입력 형태: {input_details[0]['shape']}")
        print(f"   입력 타입: {input_details[0]['dtype']}")
        print(f"   출력 형태: {output_details[0]['shape']}")
        print(f"   출력 타입: {output_details[0]['dtype']}")
        
        # 추론 테스트
        if test_data is not None:
            # 배치 차원 추가: (60, 6) → (1, 60, 6)
            input_data = np.expand_dims(test_data, axis=0).astype(np.float32)
            
            # 추론 수행
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # 결과 가져오기
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = float(output_data[0][0])
            
            print(f"✅ 추론 테스트 성공")
            print(f"   예측 확률: {prediction:.6f}")
            print(f"   예측 결과: {'보행' if prediction > 0.5 else '비보행'}")
            
            return True
        else:
            print("⚠️  테스트 데이터가 없어 추론 테스트 생략")
            return True
            
    except Exception as e:
        print(f"❌ TFLite 모델 테스트 실패: {e}")
        return False

def test_processing_pipeline():
    """전체 처리 파이프라인 테스트"""
    print("\n🔄 전체 파이프라인 테스트 중...")
    
    try:
        # 시뮬레이션 센서 데이터 생성 (실제 센서 데이터와 유사)
        # 가속도: ±2g, 자이로: ±250deg/s 범위
        accel_data = np.random.uniform(-2, 2, (60, 3))  # 가속도 3축
        gyro_data = np.random.uniform(-250, 250, (60, 3))  # 자이로 3축
        sensor_data = np.concatenate([accel_data, gyro_data], axis=1)
        
        print(f"📊 시뮬레이션 센서 데이터: {sensor_data.shape}")
        print(f"   가속도 범위: {accel_data.min():.3f} ~ {accel_data.max():.3f} m/s²")
        print(f"   자이로 범위: {gyro_data.min():.3f} ~ {gyro_data.max():.3f} deg/s")
        
        # 1단계: 스케일링
        with open("scalers/gait/minmax_scaler.pkl", 'rb') as f:
            minmax_scaler = pickle.load(f)
        with open("scalers/gait/standard_scaler.pkl", 'rb') as f:
            standard_scaler = pickle.load(f)
        
        data_reshaped = sensor_data.reshape(-1, 6)
        data_minmax = minmax_scaler.transform(data_reshaped)
        data_scaled = standard_scaler.transform(data_minmax)
        data_scaled = data_scaled.reshape(60, 6)
        
        # 2단계: TFLite 추론
        import tensorflow as tf
        interpreter = tf.lite.Interpreter("models/gait_detect/results/gait_detection.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = np.expand_dims(data_scaled, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output_data[0][0])
        
        print(f"✅ 전체 파이프라인 테스트 성공")
        print(f"   최종 예측: {prediction:.6f} ({'보행' if prediction > 0.5 else '비보행'})")
        
        return True
        
    except Exception as e:
        print(f"❌ 파이프라인 테스트 실패: {e}")
        return False

def generate_summary_report():
    """테스트 결과 요약 보고서 생성"""
    print("\n" + "="*60)
    print("📋 테스트 결과 요약")
    print("="*60)
    
    # 현재 시간
    print(f"⏰ 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 환경 정보
    print(f"🐍 Python 버전: {sys.version.split()[0]}")
    
    # 패키지 버전 확인
    try:
        import numpy
        print(f"📊 NumPy: {numpy.__version__}")
    except:
        print("📊 NumPy: 미설치")
    
    try:
        import scipy
        print(f"🔬 SciPy: {scipy.__version__}")
    except:
        print("🔬 SciPy: 미설치")
    
    try:
        import sklearn
        print(f"🧠 scikit-learn: {sklearn.__version__}")
    except:
        print("🧠 scikit-learn: 미설치")
    
    try:
        import tensorflow as tf
        print(f"🤖 TensorFlow: {tf.__version__}")
    except:
        try:
            import tflite_runtime
            print(f"🤖 TFLite Runtime: 설치됨")
        except:
            print("🤖 TensorFlow/TFLite: 미설치")
    
    print("="*60)

def main():
    """메인 테스트 함수"""
    print("🚀 보행 감지 시스템 테스트 시작")
    print("="*60)
    
    test_results = []
    
    # 1. 파일 존재 확인
    files_ok = test_file_existence()
    test_results.append(("파일 존재 확인", files_ok))
    
    if not files_ok:
        print("\n❌ 필수 파일이 누락되어 테스트를 중단합니다.")
        return False
    
    # 2. 스케일러 테스트
    scaled_data = test_scalers()
    test_results.append(("스케일러 테스트", scaled_data is not None))
    
    # 3. 메타데이터 테스트
    metadata = test_metadata()
    test_results.append(("메타데이터 테스트", metadata is not None))
    
    # 4. TFLite 모델 테스트
    model_ok = test_tflite_model(scaled_data)
    test_results.append(("TFLite 모델 테스트", model_ok))
    
    # 5. 전체 파이프라인 테스트
    if all([scaled_data is not None, model_ok]):
        pipeline_ok = test_processing_pipeline()
        test_results.append(("전체 파이프라인 테스트", pipeline_ok))
    else:
        print("\n⚠️  이전 테스트 실패로 파이프라인 테스트 생략")
        test_results.append(("전체 파이프라인 테스트", False))
    
    # 결과 요약
    generate_summary_report()
    
    print("\n📊 테스트 결과:")
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("   라즈베리파이에서 실시간 보행 감지를 사용할 준비가 되었습니다.")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다.")
        print("   README_usage_guide.md를 참고하여 문제를 해결해주세요.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 