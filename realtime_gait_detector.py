#!/usr/bin/env python3
"""
# MODIFIED [2025-06-13]: 라즈베리파이 실시간 보행 감지 시스템 구현

라즈베리파이에서 실시간으로 센서 데이터를 수집하고 전처리하여 
TFLite 모델을 통해 보행을 감지하는 완전한 시스템

기능:
- 스케일러 로드 및 실시간 데이터 전처리
- 버터워스 필터링 
- 슬라이딩 윈도우 생성
- TFLite 모델 추론
- 실시간 결과 출력
"""

import os
import numpy as np
import pickle
import json
import time
import threading
from collections import deque
from datetime import datetime
from scipy import signal
import tensorflow as tf
import logging

class GaitDetectorConfig:
    """설정 클래스"""
    # 파일 경로 설정 (실제 경로에 맞게 수정)
    SCALERS_DIR = "scalers/gait"
    MODELS_DIR = "models/gait_detect/saved_model"
    
    MINMAX_SCALER_PATH = os.path.join(SCALERS_DIR, "minmax_scaler.pkl")
    STANDARD_SCALER_PATH = os.path.join(SCALERS_DIR, "standard_scaler.pkl")
    METADATA_PATH = os.path.join(SCALERS_DIR, "metadata.json")
    TFLITE_MODEL_PATH = os.path.join("models/gait_detect/results", "gait_detection.tflite")
    
    # 센서 및 처리 설정
    SAMPLING_RATE = 30  # 30Hz
    WINDOW_SIZE = 60    # 60 프레임 (2초)
    STRIDE = 30         # 30 프레임 오버랩
    SENSOR_FEATURES = 6 # 가속도 3축 + 자이로 3축
    
    # 필터 설정 (preprocessing.py와 동일)
    FILTER_ORDER = 4
    CUTOFF_FREQ = 10
    
    # 임계값 설정
    GAIT_THRESHOLD = 0.5
    
    # 로깅 설정
    LOG_LEVEL = logging.INFO
    LOG_FILE = "gait_detection.log"

class ScalerProcessor:
    """스케일러 로딩 및 데이터 변환 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.minmax_scaler = None
        self.standard_scaler = None
        self.metadata = None
        self.load_scalers()
        
    def load_scalers(self):
        """스케일러 및 메타데이터 로드"""
        try:
            # MinMaxScaler 로드
            if os.path.exists(self.config.MINMAX_SCALER_PATH):
                with open(self.config.MINMAX_SCALER_PATH, 'rb') as f:
                    self.minmax_scaler = pickle.load(f)
                logging.info("MinMaxScaler 로드 완료")
            else:
                raise FileNotFoundError(f"MinMaxScaler 파일을 찾을 수 없습니다: {self.config.MINMAX_SCALER_PATH}")
            
            # StandardScaler 로드
            if os.path.exists(self.config.STANDARD_SCALER_PATH):
                with open(self.config.STANDARD_SCALER_PATH, 'rb') as f:
                    self.standard_scaler = pickle.load(f)
                logging.info("StandardScaler 로드 완료")
            else:
                raise FileNotFoundError(f"StandardScaler 파일을 찾을 수 없습니다: {self.config.STANDARD_SCALER_PATH}")
            
            # 메타데이터 로드
            if os.path.exists(self.config.METADATA_PATH):
                if self.config.METADATA_PATH.endswith('.json'):
                    with open(self.config.METADATA_PATH, 'r') as f:
                        self.metadata = json.load(f)
                else:  # .pkl 파일
                    with open(self.config.METADATA_PATH, 'rb') as f:
                        self.metadata = pickle.load(f)
                logging.info("메타데이터 로드 완료")
                
        except Exception as e:
            logging.error(f"스케일러 로드 오류: {e}")
            raise
    
    def transform(self, data):
        """
        데이터에 스케일링 적용 (preprocessing.py와 동일한 순서)
        
        Args:
            data: 형태가 (window_size, features) 또는 (batch, window_size, features)인 numpy 배열
            
        Returns:
            스케일링된 데이터
        """
        if self.minmax_scaler is None or self.standard_scaler is None:
            raise ValueError("스케일러가 로드되지 않았습니다")
        
        # 원본 데이터 형태 저장
        original_shape = data.shape
        
        # 2D 배열로 변환 (배치 처리를 위해)
        if len(original_shape) == 3:  # (batch, window_size, features)
            data_reshaped = data.reshape(-1, original_shape[-1])
        elif len(original_shape) == 2:  # (window_size, features)
            data_reshaped = data.reshape(-1, original_shape[-1])
        else:
            raise ValueError(f"지원하지 않는 데이터 형태: {original_shape}")
        
        try:
            # MinMaxScaler 적용 (첫 번째)
            data_minmax = self.minmax_scaler.transform(data_reshaped)
            
            # StandardScaler 적용 (두 번째)
            data_scaled = self.standard_scaler.transform(data_minmax)
            
            # 원래 형태로 복원
            data_scaled = data_scaled.reshape(original_shape)
            
            return data_scaled
            
        except Exception as e:
            logging.error(f"데이터 변환 오류: {e}")
            raise

class RealTimeGaitDetector:
    """실시간 보행 감지 메인 클래스"""
    
    def __init__(self, config=None):
        self.config = config or GaitDetectorConfig()
        
        # 로깅 설정
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        
        # 컴포넌트 초기화
        self.scaler_processor = ScalerProcessor(self.config)
        
        logging.info("실시간 보행 감지 시스템 초기화 완료")
    
    def validate_setup(self):
        """시스템 설정 검증"""
        print("🔍 시스템 설정 검증 중...")
        
        # 파일 존재 확인
        files_to_check = [
            ("MinMaxScaler", self.config.MINMAX_SCALER_PATH),
            ("StandardScaler", self.config.STANDARD_SCALER_PATH),
            ("Metadata", self.config.METADATA_PATH),
            ("TFLite Model", self.config.TFLITE_MODEL_PATH)
        ]
        
        all_files_exist = True
        for name, path in files_to_check:
            if os.path.exists(path):
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: {path} (파일 없음)")
                all_files_exist = False
        
        if not all_files_exist:
            print("\n⚠️  일부 파일이 누락되었습니다. 경로를 확인해주세요.")
            return False
        
        print("\n🎉 모든 설정이 올바르게 구성되었습니다!")
        return True

def main():
    """메인 실행 함수"""
    print("🚀 라즈베리파이 실시간 보행 감지 시스템")
    print("=" * 50)
    
    try:
        # 시스템 초기화
        detector = RealTimeGaitDetector()
        
        # 설정 검증
        if not detector.validate_setup():
            print("❌ 시스템 설정에 문제가 있습니다. 종료합니다.")
            return
        
    except Exception as e:
        logging.error(f"메인 실행 오류: {e}")
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 