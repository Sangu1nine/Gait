#!/usr/bin/env python3
"""
Gait Detection System Test Script

Tests if the preprocessed scalers and TFLite model work correctly.
"""

import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime

def test_file_existence():
    """Check existence of required files"""
    print("🔍 Checking file existence...")
    
    required_files = {
        "MinMax Scaler": "scalers/gait_detection/minmax_scaler.pkl",
        "Standard Scaler": "scalers/gait_detection/standard_scaler.pkl", 
        "Metadata": "scalers/gait_detection/metadata.json",
        "TFLite Model": "models/gait_detection/results/gait_detection.tflite"
    }
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"✅ {name}: {path} ({file_size:,} bytes)")
        else:
            print(f"❌ {name}: {path} (file not found)")
            all_exist = False
    
    return all_exist

def test_scalers():
    """Load and test scalers"""
    print("\n🔧 Testing scalers...")
    
    try:
        # Load scalers
        with open("scalers/gait_detection/minmax_scaler.pkl", 'rb') as f:
            minmax_scaler = pickle.load(f)
        print("✅ MinMaxScaler loaded successfully")
        
        with open("scalers/gait_detection/standard_scaler.pkl", 'rb') as f:
            standard_scaler = pickle.load(f)
        print("✅ StandardScaler loaded successfully")
        
        # Generate test data (60 frames, 6-axis sensor)
        test_data = np.random.randn(60, 6) * 10  # Random sensor data
        print(f"📊 Test data generated: {test_data.shape}")
        print(f"   Original range: {test_data.min():.3f} ~ {test_data.max():.3f}")
        
        # Apply scaling (same order as preprocessing.py)
        data_reshaped = test_data.reshape(-1, 6)
        
        # MinMax scaling
        data_minmax = minmax_scaler.transform(data_reshaped)
        data_minmax = data_minmax.reshape(60, 6)
        print(f"   After MinMax: {data_minmax.min():.3f} ~ {data_minmax.max():.3f}")
        
        # Standard scaling
        data_minmax_reshaped = data_minmax.reshape(-1, 6)
        data_scaled = standard_scaler.transform(data_minmax_reshaped)
        data_scaled = data_scaled.reshape(60, 6)
        print(f"   After Standard: {data_scaled.min():.3f} ~ {data_scaled.max():.3f}")
        
        print("✅ Scaler test successful")
        return data_scaled
        
    except Exception as e:
        print(f"❌ Scaler test failed: {e}")
        return None

def test_metadata():
    """Load and check metadata"""
    print("\n📋 Checking metadata...")
    
    try:
        # Try JSON format
        if os.path.exists("scalers/gait_detection/metadata.json"):
            with open("scalers/gait_detection/metadata.json", 'r') as f:
                metadata = json.load(f)
        else:
            # Try PKL format
            with open("scalers/gait_detection/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
        
        print("✅ Metadata loaded successfully")
        print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"   Stage1 data shape: {metadata.get('stage1_shape', 'N/A')}")
        print(f"   Walking subjects: {metadata.get('n_walking_subjects', 'N/A')}")
        print(f"   Non-walking subjects: {metadata.get('n_non_walking_subjects', 'N/A')}")
        
        if 'stage1_label_distribution' in metadata:
            label_dist = metadata['stage1_label_distribution']
            print(f"   Label distribution: {label_dist}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Metadata loading failed: {e}")
        return None

def test_tflite_model(test_data):
    """Test TFLite model"""
    print("\n🤖 Testing TFLite model...")
    
    try:
        # Try importing TensorFlow Lite - 라즈베리파이 환경 고려
        tflite_available = False
        tf_available = False
        
        # 1. TensorFlow 먼저 시도 (라즈베리파이에서 더 안정적)
        try:
            import tensorflow as tf
            tf_available = True
            print("✅ Using TensorFlow")
        except ImportError:
            pass
        
        # 2. TFLite Runtime 시도
        if not tf_available:
            try:
                import tflite_runtime.interpreter as tflite
                tf = type('tf', (), {'lite': type('lite', (), {'Interpreter': tflite.Interpreter})()})()
                tflite_available = True
                print("✅ Using TFLite Runtime")
            except ImportError:
                pass
        
        # 둘 다 없으면 오류
        if not tf_available and not tflite_available:
            print("❌ Neither TensorFlow nor TFLite Runtime is available")
            return False
        
        # Load model
        model_path = "models/gait_detection/results/gait_detection.tflite"
        if tf_available:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        
        # Check input/output information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TFLite model loaded successfully")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
        # Inference test
        if test_data is not None:
            # Add batch dimension: (60, 6) → (1, 60, 6)
            input_data = np.expand_dims(test_data, axis=0).astype(np.float32)
            
            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get results
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = float(output_data[0][0])
            
            print(f"✅ Inference test successful")
            print(f"   Prediction probability: {prediction:.6f}")
            print(f"   Prediction result: {'Walking' if prediction > 0.5 else 'Non-walking'}")
            
            return True
        else:
            print("⚠️  Inference test skipped due to no test data")
            return True
            
    except Exception as e:
        print(f"❌ TFLite model test failed: {e}")
        return False

def test_processing_pipeline():
    """Test complete processing pipeline"""
    print("\n🔄 Testing complete pipeline...")
    
    try:
        # Generate simulated sensor data (similar to real sensor data)
        # Accelerometer: ±2g, Gyroscope: ±250deg/s range
        accel_data = np.random.uniform(-2, 2, (60, 3))  # 3-axis accelerometer
        gyro_data = np.random.uniform(-250, 250, (60, 3))  # 3-axis gyroscope
        sensor_data = np.concatenate([accel_data, gyro_data], axis=1)
        
        print(f"📊 Simulated sensor data: {sensor_data.shape}")
        print(f"   Accelerometer range: {accel_data.min():.3f} ~ {accel_data.max():.3f} m/s²")
        print(f"   Gyroscope range: {gyro_data.min():.3f} ~ {gyro_data.max():.3f} deg/s")
        
        # Step 1: Scaling
        with open("scalers/gait_detection/minmax_scaler.pkl", 'rb') as f:
            minmax_scaler = pickle.load(f)
        with open("scalers/gait_detection/standard_scaler.pkl", 'rb') as f:
            standard_scaler = pickle.load(f)
        
        data_reshaped = sensor_data.reshape(-1, 6)
        data_minmax = minmax_scaler.transform(data_reshaped)
        data_scaled = standard_scaler.transform(data_minmax)
        data_scaled = data_scaled.reshape(60, 6)
        
        # Step 2: TFLite inference
        # 라즈베리파이 환경에 맞는 TensorFlow/TFLite 임포트
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter("models/gait_detection/results/gait_detection.tflite")
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter("models/gait_detection/results/gait_detection.tflite")
            except ImportError:
                raise ImportError("Neither TensorFlow nor TFLite Runtime is available")
        
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = np.expand_dims(data_scaled, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output_data[0][0])
        
        print(f"✅ Complete pipeline test successful")
        print(f"   Final prediction: {prediction:.6f} ({'Walking' if prediction > 0.5 else 'Non-walking'})")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def generate_summary_report():
    """Generate test result summary report"""
    print("\n" + "="*60)
    print("📋 Test Result Summary")
    print("="*60)
    
    # Current time
    print(f"⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Environment information
    print(f"🐍 Python version: {sys.version.split()[0]}")
    
    # Check package versions
    try:
        import numpy
        print(f"📊 NumPy: {numpy.__version__}")
    except:
        print("📊 NumPy: Not installed")
    
    try:
        import scipy
        print(f"🔬 SciPy: {scipy.__version__}")
    except:
        print("🔬 SciPy: Not installed")
    
    try:
        import sklearn
        print(f"🧠 scikit-learn: {sklearn.__version__}")
    except:
        print("🧠 scikit-learn: Not installed")
    
    # 라즈베리파이 환경: TensorFlow와 TFLite Runtime 둘 다 확인
    try:
        import tensorflow as tf
        print(f"🤖 TensorFlow: {tf.__version__}")
        # TFLite Runtime도 별도 확인
        try:
            import tflite_runtime
            print(f"🔧 TFLite Runtime: Available (with TensorFlow)")
        except:
            print(f"🔧 TFLite Runtime: Not separately installed")
    except:
        try:
            import tflite_runtime
            print(f"🤖 TensorFlow: Not installed")
            print(f"🔧 TFLite Runtime: Available (standalone)")
        except:
            print("🤖 TensorFlow/TFLite: Not installed")
    
    print("="*60)

def main():
    """Main test function"""
    print("🚀 Starting Gait Detection System Test")
    print("="*60)
    
    test_results = []
    
    # 1. Check file existence
    files_ok = test_file_existence()
    test_results.append(("File existence check", files_ok))
    
    if not files_ok:
        print("\n❌ Required files are missing. Stopping test.")
        return False
    
    # 2. Test scalers
    scaled_data = test_scalers()
    test_results.append(("Scaler test", scaled_data is not None))
    
    # 3. Test metadata
    metadata = test_metadata()
    test_results.append(("Metadata test", metadata is not None))
    
    # 4. Test TFLite model
    model_ok = test_tflite_model(scaled_data)
    test_results.append(("TFLite model test", model_ok))
    
    # 5. Test complete pipeline
    if all([scaled_data is not None, model_ok]):
        pipeline_ok = test_processing_pipeline()
        test_results.append(("Complete pipeline test", pipeline_ok))
    else:
        print("\n⚠️  Pipeline test skipped due to previous test failures")
        test_results.append(("Complete pipeline test", False))
    
    # Result summary
    generate_summary_report()
    
    print("\n📊 Test Results:")
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
        print("   Ready to use real-time gait detection on Raspberry Pi.")
    else:
        print("\n⚠️  Some tests failed.")
        print("   Please refer to README_usage_guide.md to resolve issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 