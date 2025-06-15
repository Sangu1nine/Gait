#!/usr/bin/env python3
"""
Gait Detection Debug Script
코드의 문제점을 진단하기 위한 디버깅 스크립트
"""

import numpy as np
import pickle
import json
import tensorflow as tf

def debug_scaler(scaler_path):
    """스케일러 정보 확인"""
    print("=" * 50)
    print("🔍 SCALER DEBUG")
    print("=" * 50)
    
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        print(f"✅ Scaler loaded successfully")
        print(f"📊 Scaler type: {type(scaler)}")
        
        if hasattr(scaler, 'mean_'):
            print(f"📈 Mean shape: {scaler.mean_.shape}")
            print(f"📈 Mean values: {scaler.mean_}")
        
        if hasattr(scaler, 'scale_'):
            print(f"📊 Scale shape: {scaler.scale_.shape}")
            print(f"📊 Scale values: {scaler.scale_}")
            
        # 스케일러가 6개 피처를 예상하는지 확인
        if hasattr(scaler, 'n_features_in_'):
            print(f"🎯 Expected features: {scaler.n_features_in_}")
            
        return scaler
        
    except Exception as e:
        print(f"❌ Scaler load error: {e}")
        return None

def debug_model(model_path):
    """모델 정보 확인"""
    print("=" * 50)
    print("🔍 MODEL DEBUG")
    print("=" * 50)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully")
        print(f"📥 Input shape: {input_details[0]['shape']}")
        print(f"📥 Input dtype: {input_details[0]['dtype']}")
        print(f"📤 Output shape: {output_details[0]['shape']}")
        print(f"📤 Output dtype: {output_details[0]['dtype']}")
        
        return interpreter, input_details, output_details
        
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return None, None, None

def test_scaling_process(scaler, window_size=60, n_features=6):
    """스케일링 과정 테스트"""
    print("=" * 50)
    print("🔍 SCALING PROCESS TEST")
    print("=" * 50)
    
    # 가짜 데이터 생성 (정상적인 범위)
    np.random.seed(42)  # 재현 가능한 결과
    fake_data = []
    for i in range(window_size):
        # 가속도계: -10 ~ +10 m/s²
        accel = np.random.uniform(-10, 10, 3)
        # 자이로스코프: -200 ~ +200 °/s
        gyro = np.random.uniform(-200, 200, 3)
        fake_data.append(list(accel) + list(gyro))
    
    print(f"📊 Generated {len(fake_data)} samples with {len(fake_data[0])} features each")
    
    # 원본 데이터 통계
    raw_array = np.array(fake_data)
    print(f"🎯 Raw data shape: {raw_array.shape}")
    print(f"📈 Raw data mean: {np.mean(raw_array, axis=0)}")
    print(f"📊 Raw data std: {np.std(raw_array, axis=0)}")
    
    if scaler is None:
        print("❌ No scaler available for testing")
        return fake_data
    
    # 3D 변환
    X_3d = raw_array.reshape(1, window_size, n_features)
    print(f"🔄 3D shape: {X_3d.shape}")
    
    # 2D 변환
    X_2d = X_3d.reshape(-1, n_features)
    print(f"🔄 2D shape: {X_2d.shape}")
    
    # 스케일링 적용
    try:
        X_scaled_2d = scaler.transform(X_2d)
        print(f"✅ Scaling successful")
        print(f"📈 Scaled 2D mean: {np.mean(X_scaled_2d, axis=0)}")
        print(f"📊 Scaled 2D std: {np.std(X_scaled_2d, axis=0)}")
        
        # 3D로 복원
        X_scaled_3d = X_scaled_2d.reshape(X_3d.shape)
        print(f"🔄 Final 3D shape: {X_scaled_3d.shape}")
        
        return fake_data, X_scaled_3d
        
    except Exception as e:
        print(f"❌ Scaling error: {e}")
        return fake_data, None

def test_model_prediction(interpreter, input_details, output_details, X_scaled):
    """모델 예측 테스트"""
    print("=" * 50)
    print("🔍 MODEL PREDICTION TEST")
    print("=" * 50)
    
    if interpreter is None or X_scaled is None:
        print("❌ Cannot test prediction - missing components")
        return
    
    try:
        # 여러 번 예측해보기 (다른 랜덤 데이터로)
        for test_num in range(3):
            print(f"\n🎯 Test #{test_num + 1}")
            
            # 매번 다른 랜덤 데이터 생성
            np.random.seed(test_num)
            test_data = np.random.normal(0, 1, (1, 60, 6)).astype(np.float32)
            
            # 예측
            interpreter.set_tensor(input_details[0]['index'], test_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"📤 Raw output: {output}")
            print(f"📤 Output shape: {output.shape}")
            print(f"📤 Output type: {type(output[0][0])}")
            
            # 확률값이 정상 범위인지 확인
            prob = output[0][0]
            if prob < 0 or prob > 1:
                print(f"⚠️ WARNING: Probability {prob} is outside [0,1] range!")
            elif prob == 0.0 or prob == 1.0:
                print(f"⚠️ WARNING: Probability is exactly {prob} - might indicate a problem!")
            else:
                print(f"✅ Probability {prob:.6f} looks normal")
                
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def debug_y_axis_issue():
    """Y축 음수 부호 문제 분석"""
    print("=" * 50)
    print("🔍 Y-AXIS NEGATIVE SIGN DEBUG")
    print("=" * 50)
    
    print("현재 코드에서 Y축 가속도에 음수 부호를 적용하고 있습니다:")
    print("accel_y = -self.accel_ms2(self.read_data(self.register_accel_yout_h))")
    print("")
    print("이것이 필요한 이유를 확인해야 합니다:")
    print("1. 센서 방향 때문인가?")
    print("2. 좌표계 변환 때문인가?")
    print("3. 훈련 데이터와 맞추기 위해서인가?")
    print("")
    print("💡 추천: 음수 부호를 제거하고 테스트해보세요")

def main():
    """메인 함수"""
    print("🔧 GAIT DETECTION DEBUG TOOL")
    print("=" * 50)
    
    # 파일 경로 (실제 파일 경로에 맞게 수정하세요)
    model_path = "models/gait_detection/model.tflite"
    scaler_path = "scalers/gait_detection/standard_scaler.pkl"
    
    # 1. 스케일러 디버그
    scaler = debug_scaler(scaler_path)
    
    # 2. 모델 디버그
    interpreter, input_details, output_details = debug_model(model_path)
    
    # 3. 스케일링 과정 테스트
    fake_data, X_scaled = test_scaling_process(scaler)
    
    # 4. 모델 예측 테스트
    test_model_prediction(interpreter, input_details, output_details, X_scaled)
    
    # 5. Y축 문제 분석
    debug_y_axis_issue()
    
    print("=" * 50)
    print("🎯 DEBUG COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()