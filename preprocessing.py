import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from datetime import datetime
import glob
import json

class GaitDataPreprocessor:
    def __init__(self, base_path="/content/drive/MyDrive/KFall_dataset/data/gait_data"):
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 버터워스 필터 설정 (4차, 10Hz 로우패스)
        self.fs = 30  # 샘플링 주파수 30Hz
        self.cutoff = 10  # 컷오프 주파수 10Hz
        self.order = 4
        
        # 스케일러
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        
        # 경로 확인 및 출력
        print(f"📁 Base path: {self.base_path}")
        print(f"📁 Walking data path: {os.path.join(self.base_path, 'walking_data')}")
        print(f"📁 Non-walking data path: {os.path.join(self.base_path, 'Selected_non_30hz')}")
        print(f"📁 Label data path: {os.path.join(self.base_path, 'support_label_data')}")
        
    def butter_lowpass_filter(self, data):
        """버터워스 로우패스 필터 적용"""
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def load_walking_data(self):
        """walking_data 로드 (SA01~SA04)"""
        walking_data = []
        walking_labels = []
        walking_subjects = []
        
        print("\n🚶 Walking 데이터 로딩 중...")
        
        for subj in range(1, 5):  # SA01~SA04
            subj_path = os.path.join(self.base_path, f"walking_data/SA{subj:02d}")
            print(f"  📂 {subj_path} 확인 중...")
            
            if not os.path.exists(subj_path):
                print(f"    ❌ 경로 없음: {subj_path}")
                continue
                
            csv_files = glob.glob(os.path.join(subj_path, "S*.csv"))
            print(f"    📄 {len(csv_files)}개 CSV 파일 발견")
            
            file_count = 0
            for csv_file in csv_files:
                try:
                    # 센서 데이터 로드
                    df = pd.read_csv(csv_file)
                    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
                    
                    # 센서 컬럼 존재 확인
                    missing_cols = [col for col in sensor_cols if col not in df.columns]
                    if missing_cols:
                        print(f"    ⚠️  {os.path.basename(csv_file)}: 센서 컬럼 누락 {missing_cols}")
                        continue
                        
                    sensor_data = df[sensor_cols].values
                    
                    # 라벨 데이터 로드
                    filename = os.path.basename(csv_file).replace('.csv', '')
                    label_file = os.path.join(self.base_path, f"support_label_data/SA{subj:02d}/{filename}_support_labels.csv")
                    
                    if os.path.exists(label_file):
                        label_df = pd.read_csv(label_file)
                        
                        # 프레임별 라벨 생성
                        frame_labels = np.full(len(sensor_data), 'non_gait', dtype=object)
                        
                        for _, row in label_df.iterrows():
                            start = int(row['start_frame']) - 1  # 0-indexed
                            end = int(row['end_frame'])
                            if start >= 0 and end <= len(sensor_data):
                                frame_labels[start:end] = row['phase']
                        
                        walking_data.append(sensor_data)
                        walking_labels.append(frame_labels)
                        walking_subjects.append(f"SA{subj:02d}")
                        file_count += 1
                        
                    else:
                        print(f"    ⚠️  라벨 파일 없음: {label_file}")
                        
                except Exception as e:
                    print(f"    ❌ 파일 로드 오류 {os.path.basename(csv_file)}: {str(e)}")
                    
            print(f"    ✅ SA{subj:02d}: {file_count}개 파일 로드 완료")
                    
        print(f"🚶 총 {len(walking_data)}개 walking 파일 로드 완료")
        return walking_data, walking_labels, walking_subjects
    
    def load_non_walking_data(self):
        """non-walking 데이터 로드 (SA06~SA38, SA34 제외)"""
        non_walking_data = []
        non_walking_subjects = []
        
        print("\n🏃 Non-walking 데이터 로딩 중...")
        
        total_files = 0
        for subj in range(6, 39):
            if subj == 34:  # SA34 제외
                continue
                
            subj_path = os.path.join(self.base_path, f"Selected_non_30hz/SA{subj:02d}")
            
            if not os.path.exists(subj_path):
                print(f"  ❌ 경로 없음: {subj_path}")
                continue
                
            csv_files = glob.glob(os.path.join(subj_path, "S*.csv"))
            file_count = 0
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
                    
                    # 센서 컬럼 존재 확인
                    missing_cols = [col for col in sensor_cols if col not in df.columns]
                    if missing_cols:
                        print(f"    ⚠️  {os.path.basename(csv_file)}: 센서 컬럼 누락 {missing_cols}")
                        continue
                        
                    sensor_data = df[sensor_cols].values
                    
                    non_walking_data.append(sensor_data)
                    non_walking_subjects.append(f"SA{subj:02d}")
                    file_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 파일 로드 오류 {os.path.basename(csv_file)}: {str(e)}")
                    
            if file_count > 0:
                print(f"  ✅ SA{subj:02d}: {file_count}개 파일 로드")
            total_files += file_count
                
        print(f"🏃 총 {len(non_walking_data)}개 non-walking 파일 로드 완료")
        return non_walking_data, non_walking_subjects
    
    def apply_filtering(self, data_list):
        """모든 데이터에 버터워스 필터 적용"""
        if not data_list:
            print("⚠️  필터링할 데이터가 없습니다.")
            return []
            
        filtered_data = []
        for i, data in enumerate(data_list):
            if i % 100 == 0:  # 진행률 표시
                print(f"  📊 필터링 진행률: {i+1}/{len(data_list)}")
                
            filtered = np.zeros_like(data)
            for j in range(data.shape[1]):  # 각 채널별로 필터 적용
                filtered[:, j] = self.butter_lowpass_filter(data[:, j])
            filtered_data.append(filtered)
        return filtered_data
    
    def create_stage1_windows(self, data_list, labels_list=None):
        """Stage1용 윈도우 생성 (60 frames, stride 30)"""
        window_size = 60
        stride = 30
        
        windows = []
        window_labels = []
        
        if not data_list:
            print("⚠️  윈도우 생성할 데이터가 없습니다.")
            return np.array([]).reshape(0, window_size, 6), np.array([])
        
        for idx, data in enumerate(data_list):
            n_frames = len(data)
            
            for start in range(0, n_frames - window_size + 1, stride):
                end = start + window_size
                windows.append(data[start:end])
                
                if labels_list is not None:
                    # Stage1: gait/non_gait 이진 분류
                    window_label_seq = labels_list[idx][start:end]
                    # DS, SSR, SSL, double_support, single_support_left, single_support_right은 모두 gait로 변환
                    binary_labels = ['gait' if l in ['DS', 'SSR', 'SSL', 'double_support', 'single_support_left', 'single_support_right'] else 'non_gait' 
                                   for l in window_label_seq]
                    # 윈도우의 대표 라벨 (다수결)
                    unique, counts = np.unique(binary_labels, return_counts=True)
                    majority_label = unique[np.argmax(counts)]
                    window_labels.append(majority_label)
                else:
                    # non-walking 데이터는 모두 non_gait
                    window_labels.append('non_gait')
                    
        if windows:
            return np.array(windows), np.array(window_labels)
        else:
            return np.array([]).reshape(0, window_size, 6), np.array([])
    
    def create_stage2_windows(self, data_list, labels_list):
        """Stage2용 윈도우 생성 (15 frames, stride 1)"""
        window_size = 15
        stride = 1
        
        windows = []
        window_labels = []
        
        if not data_list:
            print("⚠️  Stage2 윈도우 생성할 데이터가 없습니다.")
            return np.array([]).reshape(0, window_size, 6), np.array([])
        
        for idx, data in enumerate(data_list):
            n_frames = len(data)
            
            # 처음과 끝 15프레임 제외
            for start in range(15, n_frames - window_size - 14, stride):
                end = start + window_size
                windows.append(data[start:end])
                
                # 각 프레임의 라벨 (DS, SSR, SSL만)
                frame_labels = []
                for frame_idx in range(start, end):
                    label = labels_list[idx][frame_idx]
                    if label in ['DS', 'SSR', 'SSL', 'double_support', 'single_support_left', 'single_support_right']:
                        # 라벨명 정규화
                        if label in ['double_support', 'DS']:
                            frame_labels.append('DS')
                        elif label in ['single_support_left', 'SSL']:
                            frame_labels.append('SSL')
                        elif label in ['single_support_right', 'SSR']:
                            frame_labels.append('SSR')
                        else:
                            frame_labels.append(label)
                    else:
                        frame_labels.append('DS')  # 기본값
                
                window_labels.append(frame_labels)
                
        if windows:
            return np.array(windows), np.array(window_labels)
        else:
            return np.array([]).reshape(0, window_size, 6), np.array([])
    
    def process_and_save(self):
        """전체 전처리 프로세스 실행"""
        print("=" * 60)
        print("🚀 데이터 전처리 프로세스 시작")
        print("=" * 60)
        
        # 1. 데이터 로드
        walking_data, walking_labels, walking_subjects = self.load_walking_data()
        non_walking_data, non_walking_subjects = self.load_non_walking_data()
        
        print(f"\n📊 데이터 로드 결과:")
        print(f"  🚶 Walking 데이터: {len(walking_data)}개 파일")
        print(f"  🏃 Non-walking 데이터: {len(non_walking_data)}개 파일")
        
        # 데이터가 하나도 없으면 오류
        if len(walking_data) == 0 and len(non_walking_data) == 0:
            raise ValueError("❌ 로드된 데이터가 없습니다. 경로와 파일을 확인해주세요.")
        
        # 2. 버터워스 필터 적용
        print(f"\n🔧 버터워스 필터 적용 중...")
        if walking_data:
            walking_data = self.apply_filtering(walking_data)
            print(f"  ✅ Walking 데이터 필터링 완료")
        if non_walking_data:
            non_walking_data = self.apply_filtering(non_walking_data)
            print(f"  ✅ Non-walking 데이터 필터링 완료")
        
        # 3. Stage1 데이터 생성
        print(f"\n📦 Stage1 윈도우 생성 중...")
        walking_windows_s1, walking_labels_s1 = self.create_stage1_windows(walking_data, walking_labels)
        non_walking_windows_s1, non_walking_labels_s1 = self.create_stage1_windows(non_walking_data)
        
        print(f"  🚶 Walking windows: {walking_windows_s1.shape}")
        print(f"  🏃 Non-walking windows: {non_walking_windows_s1.shape}")
        
        # 전체 Stage1 데이터 결합 (빈 배열 처리)
        if walking_windows_s1.size > 0 and non_walking_windows_s1.size > 0:
            all_windows_s1 = np.vstack([walking_windows_s1, non_walking_windows_s1])
            all_labels_s1 = np.hstack([walking_labels_s1, non_walking_labels_s1])
        elif walking_windows_s1.size > 0:
            all_windows_s1 = walking_windows_s1
            all_labels_s1 = walking_labels_s1
        elif non_walking_windows_s1.size > 0:
            all_windows_s1 = non_walking_windows_s1
            all_labels_s1 = non_walking_labels_s1
        else:
            raise ValueError("❌ 생성된 윈도우가 없습니다.")
        
        print(f"  📦 전체 Stage1 windows: {all_windows_s1.shape}")
        
        # 4. 스케일링 (Stage1)
        print(f"\n⚖️  Stage1 데이터 스케일링 중...")
        # Reshape for scaling
        n_samples, n_frames, n_features = all_windows_s1.shape
        all_windows_s1_reshaped = all_windows_s1.reshape(-1, n_features)
        
        # Fit and transform
        all_windows_s1_minmax = self.minmax_scaler.fit_transform(all_windows_s1_reshaped)
        all_windows_s1_standard = self.standard_scaler.fit_transform(all_windows_s1_reshaped)
        
        # Reshape back
        all_windows_s1_minmax = all_windows_s1_minmax.reshape(n_samples, n_frames, n_features)
        all_windows_s1_standard = all_windows_s1_standard.reshape(n_samples, n_frames, n_features)
        
        print(f"  ✅ 스케일링 완료: MinMax, Standard")
        
        # 5. Stage2 데이터 생성 (walking data만)
        print(f"\n📦 Stage2 윈도우 생성 중...")
        if walking_data:
            walking_windows_s2, walking_labels_s2 = self.create_stage2_windows(walking_data, walking_labels)
            print(f"  📦 Stage2 windows: {walking_windows_s2.shape}")
        else:
            walking_windows_s2 = np.array([]).reshape(0, 15, 6)
            walking_labels_s2 = np.array([])
            print(f"  ⚠️  Walking 데이터가 없어 Stage2 윈도우 생성 불가")
        
        # 6. 저장
        save_dir = os.path.join(self.base_path, f"preprocessed_{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 데이터 저장 중: {save_dir}")
        
        # Stage1 데이터 저장
        np.save(os.path.join(save_dir, "stage1_data_minmax.npy"), all_windows_s1_minmax)
        np.save(os.path.join(save_dir, "stage1_data_standard.npy"), all_windows_s1_standard)
        np.save(os.path.join(save_dir, "stage1_labels.npy"), all_labels_s1)
        
        # Stage2 데이터 저장
        if walking_windows_s2.size > 0:
            np.save(os.path.join(save_dir, "stage2_data.npy"), walking_windows_s2)
            np.save(os.path.join(save_dir, "stage2_labels.npy"), walking_labels_s2)
        
        # 스케일러 저장
        with open(os.path.join(save_dir, "minmax_scaler.pkl"), 'wb') as f:
            pickle.dump(self.minmax_scaler, f)
        with open(os.path.join(save_dir, "standard_scaler.pkl"), 'wb') as f:
            pickle.dump(self.standard_scaler, f)
            
        # 메타데이터 저장
        metadata = {
            'timestamp': self.timestamp,
            'n_walking_subjects': len(set(walking_subjects)) if walking_subjects else 0,
            'n_non_walking_subjects': len(set(non_walking_subjects)) if non_walking_subjects else 0,
            'stage1_shape': all_windows_s1_minmax.shape,
            'stage2_shape': walking_windows_s2.shape,
            'walking_subjects': list(set(walking_subjects)) if walking_subjects else [],
            'non_walking_subjects': list(set(non_walking_subjects)) if non_walking_subjects else [],
            'stage1_label_distribution': {label: int(count) for label, count in zip(*np.unique(all_labels_s1, return_counts=True))},
            'total_files_processed': {
                'walking': len(walking_data),
                'non_walking': len(non_walking_data)
            }
        }
        
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        # 메타데이터를 JSON으로도 저장 (가독성을 위해)
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print("=" * 60)
        print("✅ 전처리 완료!")
        print("=" * 60)
        print(f"📊 최종 결과:")
        print(f"  📦 Stage1 데이터: {all_windows_s1_minmax.shape}")
        print(f"  📦 Stage2 데이터: {walking_windows_s2.shape}")
        print(f"  📋 라벨 분포: {metadata['stage1_label_distribution']}")
        print(f"  💾 저장 위치: {save_dir}")
        print("=" * 60)
        
        return save_dir

if __name__ == "__main__":
    preprocessor = GaitDataPreprocessor()
    save_path = preprocessor.process_and_save()
    print(f"전처리된 데이터가 저장되었습니다: {save_path}")