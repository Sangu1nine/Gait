import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import re
import pickle
warnings.filterwarnings('ignore')

class FallDataProcessor:
    def __init__(self, base_path, sampling_rate=100, window_size=150, stride=75):
        self.base_path = base_path
        self.sampling_rate = sampling_rate    # 샘플링 레이트 (Hz)
        self.window_size = window_size        # 윈도우 크기
        self.stride = stride                  # 스트라이드

        # 데이터 경로 설정
        self.fall_data_path = os.path.join(base_path, 'extracted_data')
        self.nonfall_data_path = os.path.join(base_path, 'selected_tasks_data')
        self.label_data_path = os.path.join(base_path, 'label_data_new')
        self.output_path = os.path.join(base_path, 'preprocessed_data_stride50_0508')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_path, exist_ok=True)

        # 데이터 저장 변수
        self.features = []
        self.labels = []

        # 사용할 센서 특성 정의 (가속도 3축, 각가속도 3축)
        self.sensor_features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']

        # Scaler 딕셔너리 - 각 센서별로 개별 스케일러 저장
        self.minmax_scalers = {feature: MinMaxScaler() for feature in self.sensor_features}
        self.standard_scalers = {feature: StandardScaler() for feature in self.sensor_features}

    def load_and_process_falls(self):
        """낙상 데이터 처리 (extracted_data 폴더)"""
        print("\n낙상 데이터 처리 시작...")
        subject_folders = glob(os.path.join(self.fall_data_path, "SA*"))

        for subject_folder in subject_folders:
            subject_id = os.path.basename(subject_folder)
            print(f"피험자 {subject_id} 처리 중...")
            fall_files = glob(os.path.join(subject_folder, "extracted_*.csv"))

            for fall_file in fall_files:
                self._process_fall_file(fall_file)

    def _process_fall_file(self, file_path):
        """낙상 파일 처리 (모든 프레임을 낙상으로 처리)"""
        try:
            # 데이터 로드
            sensor_data = pd.read_csv(file_path)

            # 필요한 컬럼이 모두 있는지 확인
            if not all(feature in sensor_data.columns for feature in self.sensor_features):
                missing_cols = [f for f in self.sensor_features if f not in sensor_data.columns]
                print(f"  필요한 컬럼 없음: {missing_cols}")
                return

            # NaN 값 처리
            for feature in self.sensor_features:
                if sensor_data[feature].isna().any():
                    sensor_data[feature] = sensor_data[feature].fillna(method='ffill').fillna(method='bfill')

            # 모든 프레임을 낙상(1)으로 레이블링
            labels = np.ones(len(sensor_data))

            # 윈도우 추출
            self._extract_windows(sensor_data, labels)

        except Exception as e:
            print(f"  오류 발생 ({os.path.basename(file_path)}): {str(e)}")

    def _extract_windows(self, data, labels):
        """데이터 윈도우 추출"""
        if len(data) < self.window_size:
            return

        if not all(feature in data.columns for feature in self.sensor_features):
            missing_cols = [f for f in self.sensor_features if f not in data.columns]
            print(f"  윈도우 추출 불가: 필요한 컬럼 없음 {missing_cols}")
            return

        # 각 센서 특성에 대해 개별적으로 처리
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window_data = {}

            for feature in self.sensor_features:
                # 윈도우 추출
                window = data[feature].values[i:i + self.window_size]
                window_data[feature] = window

            # 윈도우의 레이블 결정
            window_labels = labels[i:i + self.window_size]
            label = np.bincount(window_labels.astype(int)).argmax()

            # 모든 센서 데이터를 하나의 윈도우로 결합
            combined_window = np.array([window_data[feature] for feature in self.sensor_features]).T

            self.features.append(combined_window)
            self.labels.append(label)

    def _apply_min_max_scaling(self, X):
        """각 센서 채널별로 MinMaxScaler 적용"""
        num_windows, window_length, num_features = X.shape
        X_scaled = np.zeros_like(X)

        # 각 특성에 대해 개별적으로 스케일링
        for i, feature in enumerate(self.sensor_features):
            # 특성 데이터 추출 (모든 윈도우, 모든 시간 스텝에서의 i번째 특성)
            feature_data = X[:, :, i].reshape(-1, 1)

            # MinMaxScaler 학습 및 변환
            scaler = self.minmax_scalers[feature]
            scaled_data = scaler.fit_transform(feature_data)

            # 스케일링된 데이터를 원래 형태로 재구성
            X_scaled[:, :, i] = scaled_data.reshape(num_windows, window_length)

        return X_scaled
    
    def _apply_standard_scaling(self, X):
        """각 센서 채널별로 StandardScaler 적용"""
        num_windows, window_length, num_features = X.shape
        X_scaled = np.zeros_like(X)

        # 각 특성에 대해 개별적으로 스케일링
        for i, feature in enumerate(self.sensor_features):
            # 특성 데이터 추출 (모든 윈도우, 모든 시간 스텝에서의 i번째 특성)
            feature_data = X[:, :, i].reshape(-1, 1)

            # StandardScaler 학습 및 변환
            scaler = self.standard_scalers[feature]
            scaled_data = scaler.fit_transform(feature_data)

            # 스케일링된 데이터를 원래 형태로 재구성
            X_scaled[:, :, i] = scaled_data.reshape(num_windows, window_length)

        return X_scaled

    def _split_and_scale(self, X, y):
        """데이터 분할 및 스케일러 적용 (MinMax와 Standard 모두 적용)"""
        # 훈련/테스트 분할 (8:2 비율)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 훈련 데이터 복사 (MinMax와 Standard 모두 적용할 것이므로)
        X_train_scaled = np.copy(X_train)
        X_test_scaled = np.copy(X_test)
        
        # 각 특성에 대해 개별적으로 처리
        for i, feature in enumerate(self.sensor_features):
            # 훈련 데이터 변환 - 먼저 MinMax 스케일링 적용
            feature_data_train = X_train[:, :, i].reshape(-1, 1)
            minmax_scaled_data = self.minmax_scalers[feature].fit_transform(feature_data_train)
            
            # 그 다음 MinMax 결과에 Standard 스케일링 적용
            standard_scaled_data = self.standard_scalers[feature].fit_transform(minmax_scaled_data)
            X_train_scaled[:, :, i] = standard_scaled_data.reshape(X_train.shape[0], X_train.shape[1])
            
            # 테스트 데이터도 같은 방식으로 변환
            feature_data_test = X_test[:, :, i].reshape(-1, 1)
            minmax_scaled_data_test = self.minmax_scalers[feature].transform(feature_data_test)
            standard_scaled_data_test = self.standard_scalers[feature].transform(minmax_scaled_data_test)
            X_test_scaled[:, :, i] = standard_scaled_data_test.reshape(X_test.shape[0], X_test.shape[1])

        return X_train_scaled, X_test_scaled, y_train, y_test

    def prepare_dataset(self):
        """데이터셋 준비"""
        # 낙상/비낙상 데이터 처리
        self.load_and_process_falls()
        self.load_and_process_non_falls()

        if len(self.features) == 0:
            print("추출된 데이터가 없습니다!")
            return None

        # 배열 변환
        X = np.array(self.features)
        y = np.array(self.labels)

        print(f"\n데이터 형태: {X.shape} (샘플 수, {self.window_size} 타임스텝, {len(self.sensor_features)} 센서 특성)")

        # 클래스 분포 출력
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print("\n클래스 분포:")
        if 0 in unique:
            idx = np.where(unique == 0)[0][0]
            print(f"  비낙상: {counts[idx]}개 ({counts[idx]/total*100:.1f}%)")
        if 1 in unique:
            idx = np.where(unique == 1)[0][0]
            print(f"  낙상: {counts[idx]}개 ({counts[idx]/total*100:.1f}%)")

        # 데이터 분할 및 스케일링
        return self._split_and_scale(X, y)

    def _is_valid_nonfall_file(self, file_path):
        """비낙상 파일 유효성 검사 (T20-T34는 낙상 데이터)"""
        try:
            file_name = os.path.basename(file_path)
            task_match = re.search(r'T(\d+)', file_name)
            if task_match:
                task_num = int(task_match.group(1))
                return not (20 <= task_num <= 34)
            return False
        except Exception as e:
            print(f"  파일명 파싱 오류 ({os.path.basename(file_path)}): {str(e)}")
            return False

    def load_and_process_non_falls(self):
        """비낙상 데이터 처리 (selected_tasks_data 폴더)"""
        print("\n비낙상 데이터 처리 시작...")
        subject_folders = glob(os.path.join(self.nonfall_data_path, "SA*"))

        for subject_folder in subject_folders:
            subject_id = os.path.basename(subject_folder)
            print(f"피험자 {subject_id} 처리 중...")

            # 모든 CSV 파일
            all_files = glob(os.path.join(subject_folder, "S*.csv"))
            print(f"  - 발견된 CSV 파일 수: {len(all_files)}")

            valid_files = []
            invalid_files = []

            # 파일 필터링 전에 모든 파일을 확인
            for file in all_files:
                filename = os.path.basename(file)
                t_match = re.search(r'T(\d+)', filename)
                if t_match:
                    task_num = int(t_match.group(1))
                    if 20 <= task_num <= 34:
                        invalid_files.append(filename)
                    else:
                        valid_files.append(file)
                else:
                    print(f"  경고: 'T' 패턴이 없는 파일: {filename}")

            print(f"  - 유효한 비낙상 파일 수: {len(valid_files)}")
            print(f"  - 제외된 파일 수 (T20-T34): {len(invalid_files)}")

            # 비낙상 파일만 처리
            for file_path in valid_files:
                self._process_nonfall_file(file_path)

    def _process_nonfall_file(self, file_path):
        """비낙상 파일 처리"""
        try:
            # 데이터 로드
            sensor_data = pd.read_csv(file_path)

            # 필요한 컬럼이 모두 있는지 확인
            if not all(feature in sensor_data.columns for feature in self.sensor_features):
                missing_cols = [f for f in self.sensor_features if f not in sensor_data.columns]
                print(f"  필요한 컬럼 없음: {missing_cols}")
                return

            # NaN 값 처리
            for feature in self.sensor_features:
                if sensor_data[feature].isna().any():
                    sensor_data[feature] = sensor_data[feature].fillna(method='ffill').fillna(method='bfill')

            # 모든 프레임을 비낙상(0)으로 레이블링
            labels = np.zeros(len(sensor_data))

            # 윈도우 추출
            self._extract_windows(sensor_data, labels)

        except Exception as e:
            print(f"  오류 발생 ({os.path.basename(file_path)}): {str(e)}")

    def save_scalers(self):
        """학습된 스케일러 저장"""
        # 스케일러 저장 디렉토리
        scalers_dir = os.path.join(self.output_path, 'scalers')
        os.makedirs(scalers_dir, exist_ok=True)
        
        # MinMaxScaler 저장
        for feature, scaler in self.minmax_scalers.items():
            scaler_path = os.path.join(scalers_dir, f"{feature}_minmax_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # StandardScaler 저장
        for feature, scaler in self.standard_scalers.items():
            scaler_path = os.path.join(scalers_dir, f"{feature}_standard_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"스케일러 저장 완료: {scalers_dir}")
        
    def save_datasets(self, X_train, X_test, y_train, y_test):
        """처리된 데이터셋 저장"""
        # 스케일링된 데이터 저장
        np.save(os.path.join(self.output_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_path, 'X_test.npy'), X_test)
        
        # 레이블 데이터 저장
        np.save(os.path.join(self.output_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_path, 'y_test.npy'), y_test)
        
        print(f"데이터셋 저장 완료: {self.output_path}")
        
    def process_and_save(self):
        """전체 처리 및 저장 프로세스"""
        print("데이터 처리 및 저장 시작...")
        
        # 데이터셋 준비
        result = self.prepare_dataset()
        
        if result is None:
            print("처리할 데이터가 없습니다.")
            return
        
        X_train, X_test, y_train, y_test = result
        
        # 스케일러 저장
        self.save_scalers()
        
        # 데이터셋 저장
        self.save_datasets(X_train, X_test, y_train, y_test)
        
        print("모든 처리 완료!")
        
# 메인 실행 코드
if __name__ == "__main__":
    base_path = "C:/gitproject/Final_Project/data/"
    processor = FallDataProcessor(base_path)
    processor.process_and_save()