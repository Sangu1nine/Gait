import os
import numpy as np
import pandas as pd
from scipy import signal
import pickle
from datetime import datetime
import glob
import json
import warnings

# 경고 필터링
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class GaitDataPreprocessor:
    def __init__(self, base_path="C:/Gait"):
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 버터워스 필터 설정
        self.fs = 30  # 샘플링 주파수 30Hz
        self.cutoff = 10  # 컷오프 주파수 10Hz
        self.order = 4
        
        # 경로 확인 및 출력
        print(f"📁 Base path: {self.base_path}")
        print(f"📁 Walking data path: {os.path.join(self.base_path, 'walking_data')}")
        print(f"📁 Non-walking data path: {os.path.join(self.base_path, 'Selected_non_30hz')}")
        print(f"📁 Label data path: {os.path.join(self.base_path, 'support_label_data')}")
        
    def butter_lowpass_filter(self, data):
        """버터워스 로우패스 필터 적용"""
        try:
            nyq = 0.5 * self.fs
            normal_cutoff = self.cutoff / nyq
            b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
            filtered_data = signal.filtfilt(b, a, data, axis=0)
            return filtered_data.astype(np.float32)
        except Exception as e:
            print(f"    ⚠️  필터 적용 오류: {str(e)}")
            return data.astype(np.float32)
    
    def extract_subject_from_filename(self, filepath):
        """파일 경로에서 피험자 ID 추출"""
        # 경로에서 피험자 폴더명 추출 (예: .../SA01/S1_L_01.csv -> SA01)
        path_parts = filepath.replace('\\', '/').split('/')
        for part in path_parts:
            if part.startswith('SA') and len(part) == 4:
                return part
        
        # 폴더명에서 찾지 못한 경우 파일명에서 추출 시도
        filename = os.path.basename(filepath)
        if filename.startswith('SA'):
            return filename[:4]
        
        return None
    
    def load_walking_data(self):
        """walking_data 로드 (실제 파일에서 피험자 ID 추출)"""
        walking_data = []
        walking_labels = []
        walking_subjects = []
        walking_filenames = []
        
        print("\n🚶 Walking 데이터 로딩 중...")
        
        # 모든 walking_data 하위 폴더 검색
        walking_base = os.path.join(self.base_path, "walking_data")
        if not os.path.exists(walking_base):
            print(f"❌ Walking data 폴더가 없습니다: {walking_base}")
            return [], [], [], []
        
        # SA*로 시작하는 모든 폴더 찾기
        subject_folders = glob.glob(os.path.join(walking_base, "SA*"))
        
        total_files = 0
        for subj_folder in sorted(subject_folders):
            subj_id = os.path.basename(subj_folder)
            print(f"  📂 {subj_id} 처리 중...")
            
            csv_files = glob.glob(os.path.join(subj_folder, "*.csv"))
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
                    
                    sensor_data = df[sensor_cols].values.astype(np.float32)
                    
                    # 라벨 데이터 로드
                    filename = os.path.basename(csv_file).replace('.csv', '')
                    label_file = os.path.join(self.base_path, f"support_label_data/{subj_id}/{filename}_support_labels.csv")
                    
                    if os.path.exists(label_file):
                        label_df = pd.read_csv(label_file)
                        
                        # 프레임별 라벨 생성
                        frame_labels = np.full(len(sensor_data), 'non_gait', dtype='U20')
                        
                        for _, row in label_df.iterrows():
                            start = max(0, int(row['start_frame']) - 1)  # 0-indexed, 음수 방지
                            end = min(len(sensor_data), int(row['end_frame']))  # 범위 초과 방지
                            if start < end:
                                frame_labels[start:end] = row['phase']
                        
                        # 실제 파일 경로에서 피험자 ID 추출
                        extracted_subject = self.extract_subject_from_filename(csv_file)
                        final_subject_id = extracted_subject if extracted_subject else subj_id
                        
                        walking_data.append(sensor_data)
                        walking_labels.append(frame_labels)
                        walking_subjects.append(final_subject_id)
                        walking_filenames.append(os.path.basename(csv_file))
                        file_count += 1
                        
                    else:
                        print(f"    ⚠️  라벨 파일 없음: {os.path.basename(label_file)}")
                        
                except Exception as e:
                    print(f"    ❌ 파일 로드 오류 {os.path.basename(csv_file)}: {str(e)}")
                    
            if file_count > 0:
                print(f"    ✅ {subj_id}: {file_count}개 파일 로드 완료")
                total_files += file_count
                    
        print(f"🚶 총 {total_files}개 walking 파일 로드 완료")
        unique_subjects = list(set(walking_subjects))
        print(f"📊 Walking 피험자: {sorted(unique_subjects)}")
        
        return walking_data, walking_labels, walking_subjects, walking_filenames
    
    def load_non_walking_data(self):
        """non-walking 데이터 로드 (실제 파일에서 피험자 ID 추출)"""
        non_walking_data = []
        non_walking_subjects = []
        non_walking_filenames = []
        
        print("\n🏃 Non-walking 데이터 로딩 중...")
        
        # 모든 Selected_non_30hz 하위 폴더 검색
        non_walking_base = os.path.join(self.base_path, "Selected_non_30hz")
        if not os.path.exists(non_walking_base):
            print(f"❌ Non-walking data 폴더가 없습니다: {non_walking_base}")
            return [], [], []
        
        # SA*로 시작하는 모든 폴더 찾기 (SA34 제외)
        subject_folders = glob.glob(os.path.join(non_walking_base, "SA*"))
        subject_folders = [f for f in subject_folders if not os.path.basename(f) == 'SA34']
        
        total_files = 0
        for subj_folder in sorted(subject_folders):
            subj_id = os.path.basename(subj_folder)
            
            csv_files = glob.glob(os.path.join(subj_folder, "*.csv"))
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
                        
                    sensor_data = df[sensor_cols].values.astype(np.float32)
                    
                    # 실제 파일 경로에서 피험자 ID 추출
                    extracted_subject = self.extract_subject_from_filename(csv_file)
                    final_subject_id = extracted_subject if extracted_subject else subj_id
                    
                    non_walking_data.append(sensor_data)
                    non_walking_subjects.append(final_subject_id)
                    non_walking_filenames.append(os.path.basename(csv_file))
                    file_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 파일 로드 오류 {os.path.basename(csv_file)}: {str(e)}")
                    
            if file_count > 0:
                print(f"  ✅ {subj_id}: {file_count}개 파일 로드")
                total_files += file_count
                
        print(f"🏃 총 {total_files}개 non-walking 파일 로드 완료")
        unique_subjects = list(set(non_walking_subjects))
        print(f"📊 Non-walking 피험자: {sorted(unique_subjects)}")
        
        return non_walking_data, non_walking_subjects, non_walking_filenames
    
    def apply_filtering(self, data_list):
        """모든 데이터에 버터워스 필터 적용"""
        if not data_list:
            print("⚠️  필터링할 데이터가 없습니다.")
            return []
            
        filtered_data = []
        for i, data in enumerate(data_list):
            if i % 100 == 0:  # 진행률 표시
                print(f"  📊 필터링 진행률: {i+1}/{len(data_list)}")
                
            filtered = np.zeros_like(data, dtype=np.float32)
            for j in range(data.shape[1]):  # 각 채널별로 필터 적용
                filtered[:, j] = self.butter_lowpass_filter(data[:, j])
            filtered_data.append(filtered)
        return filtered_data
    
    def create_stage1_windows(self, data_list, subjects_list, filenames_list, labels_list=None):
        """Stage1용 윈도우 생성 (정확한 피험자 매핑)"""
        window_size = 60
        stride = 30
        
        windows = []
        window_labels = []
        window_subjects = []
        window_file_info = []  # 어떤 파일에서 왔는지 추적
        
        if not data_list:
            print("⚠️  윈도우 생성할 데이터가 없습니다.")
            return np.array([]).reshape(0, window_size, 6), np.array([]), np.array([]), []
        
        for idx, data in enumerate(data_list):
            subject_id = subjects_list[idx]
            filename = filenames_list[idx]
            n_frames = len(data)
            
            for start in range(0, n_frames - window_size + 1, stride):
                end = start + window_size
                windows.append(data[start:end])
                window_subjects.append(subject_id)
                window_file_info.append({
                    'subject': subject_id,
                    'filename': filename,
                    'start_frame': start,
                    'end_frame': end
                })
                
                if labels_list is not None:
                    # Walking 데이터: gait phase 기반 라벨링
                    window_label_seq = labels_list[idx][start:end]
                    
                    # Gait phase들을 gait로 변환
                    gait_phases = ['DS', 'SSR', 'SSL', 'double_support', 
                                 'single_support_left', 'single_support_right']
                    binary_labels = ['gait' if l in gait_phases else 'non_gait' 
                                   for l in window_label_seq]
                    
                    # 윈도우의 대표 라벨 (50% 이상인 클래스)
                    gait_ratio = np.mean([1 if l == 'gait' else 0 for l in binary_labels])
                    majority_label = 'gait' if gait_ratio >= 0.5 else 'non_gait'
                    window_labels.append(majority_label)
                else:
                    # Non-walking 데이터는 모두 non_gait
                    window_labels.append('non_gait')
                    
        if windows:
            return (np.array(windows, dtype=np.float32), 
                   np.array(window_labels, dtype='U20'),
                   np.array(window_subjects, dtype='U20'),
                   window_file_info)
        else:
            return (np.array([]).reshape(0, window_size, 6), 
                   np.array([]), 
                   np.array([]),
                   [])
    
    def process_and_save(self):
        """전체 전처리 프로세스 실행 (Data Leakage 해결)"""
        print("=" * 60)
        print("🚀 Stage1 데이터 전처리 프로세스 시작 (Data Leakage 방지)")
        print("=" * 60)
        
        # 1. 데이터 로드
        walking_data, walking_labels, walking_subjects, walking_files = self.load_walking_data()
        non_walking_data, non_walking_subjects, non_walking_files = self.load_non_walking_data()
        
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
        
        # 3. Stage1 윈도우 생성 (정확한 피험자 매핑)
        print(f"\n📦 Stage1 윈도우 생성 중...")
        walking_windows, walking_labels_w, walking_groups, walking_info = self.create_stage1_windows(
            walking_data, walking_subjects, walking_files, walking_labels)
        non_walking_windows, non_walking_labels_w, non_walking_groups, non_walking_info = self.create_stage1_windows(
            non_walking_data, non_walking_subjects, non_walking_files)
        
        print(f"  🚶 Walking windows: {walking_windows.shape}")
        print(f"  🏃 Non-walking windows: {non_walking_windows.shape}")
        
        # 전체 Stage1 데이터 결합
        if walking_windows.size > 0 and non_walking_windows.size > 0:
            all_windows = np.vstack([walking_windows, non_walking_windows])
            all_labels = np.hstack([walking_labels_w, non_walking_labels_w])
            all_groups = np.hstack([walking_groups, non_walking_groups])
            all_info = walking_info + non_walking_info
        elif walking_windows.size > 0:
            all_windows = walking_windows
            all_labels = walking_labels_w
            all_groups = walking_groups
            all_info = walking_info
        elif non_walking_windows.size > 0:
            all_windows = non_walking_windows
            all_labels = non_walking_labels_w
            all_groups = non_walking_groups
            all_info = non_walking_info
        else:
            raise ValueError("❌ 생성된 윈도우가 없습니다.")
        
        print(f"  📦 전체 Stage1 windows: {all_windows.shape}")
        
        # 피험자 정보 검증 및 출력
        unique_subjects = np.unique(all_groups)
        print(f"  👥 총 피험자 수: {len(unique_subjects)}명")
        print(f"  👥 피험자 목록: {sorted(unique_subjects)}")
        
        # 피험자별 샘플 수 확인
        subject_counts = {}
        for subject in unique_subjects:
            count = np.sum(all_groups == subject)
            subject_counts[subject] = int(count)
        
        print(f"  📊 피험자별 윈도우 수:")
        for subject in sorted(subject_counts.keys()):
            print(f"    - {subject}: {subject_counts[subject]}개")
        
        # 4. 저장 (스케일링 제거 - 모델에서 처리)
        save_dir = os.path.join(self.base_path, f"stage1_preprocessed_{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Stage1 데이터 저장 중: {save_dir}")
        
        # 원본 데이터만 저장 (스케일링은 모델에서 수행)
        np.save(os.path.join(save_dir, "stage1_data.npy"), all_windows)
        np.save(os.path.join(save_dir, "stage1_labels.npy"), all_labels)
        np.save(os.path.join(save_dir, "stage1_groups.npy"), all_groups)
        
        # 상세 정보 저장
        with open(os.path.join(save_dir, "window_info.pkl"), 'wb') as f:
            pickle.dump(all_info, f)
        
        # 메타데이터 생성
        walking_subjects_unique = list(set(walking_subjects)) if walking_subjects else []
        non_walking_subjects_unique = list(set(non_walking_subjects)) if non_walking_subjects else []
        
        metadata = {
            'timestamp': self.timestamp,
            'processing_stage': 'stage1_only',
            'data_leakage_prevention': True,
            'scaling_applied': False,
            'scaling_note': 'Scaling should be applied in model training using GroupKFold',
            'n_walking_subjects': len(walking_subjects_unique),
            'n_non_walking_subjects': len(non_walking_subjects_unique),
            'stage1_shape': list(all_windows.shape),
            'walking_subjects': walking_subjects_unique,
            'non_walking_subjects': non_walking_subjects_unique,
            'all_subjects': list(unique_subjects),
            'stage1_label_distribution': {
                str(label): int(count) for label, count in 
                zip(*np.unique(all_labels, return_counts=True))
            },
            'subject_window_counts': subject_counts,
            'total_files_processed': {
                'walking': len(walking_data),
                'non_walking': len(non_walking_data)
            },
            'window_config': {
                'window_size': 60,
                'stride': 30,
                'sampling_rate': 30,
                'label_threshold': 0.5  # 50% 이상 gait면 gait 라벨
            },
            'filter_config': {
                'type': 'butterworth_lowpass',
                'order': 4,
                'cutoff_hz': 10,
                'sampling_rate': 30
            }
        }
        
        # 메타데이터 저장
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        # GroupKFold 사용 예시 저장 (스케일링 포함)
        example_code = '''# Data Leakage 방지를 위한 올바른 사용법
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# 데이터 로드
X = np.load("stage1_data.npy")  # 원본 데이터 (스케일링 안됨)
y = np.load("stage1_labels.npy")
groups = np.load("stage1_groups.npy")

print(f"데이터 형태: {X.shape}")
print(f"라벨 수: {len(np.unique(y))}")
print(f"피험자 수: {len(np.unique(groups))}")

# 올바른 방법: GroupKFold + 내부 스케일링
from sklearn.preprocessing import LabelEncoder

# 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# GroupKFold 설정
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
    # 데이터 분할
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # 훈련 세트로만 스케일러 학습
    scaler = StandardScaler()
    
    # 3D 데이터를 2D로 변환하여 스케일링
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    # 훈련 데이터로 fit, 훈련+테스트 데이터에 transform
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)  # fit 없이 transform만!
    
    # 다시 3D로 변환
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # 피험자 확인
    train_subjects = np.unique(groups[train_idx])
    test_subjects = np.unique(groups[test_idx])
    
    print(f"Fold {fold+1}:")
    print(f"  Train 피험자: {train_subjects}")
    print(f"  Test 피험자: {test_subjects}")
    print(f"  Train 샘플: {len(X_train_scaled)}, Test 샘플: {len(X_test_scaled)}")
    
    # 여기서 모델 학습 및 평가
    # model.fit(X_train_scaled, y_train)
    # score = model.evaluate(X_test_scaled, y_test)

# 잘못된 방법 (하지 마세요!)
# scaler = StandardScaler()
# X_all_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))  # 전체 데이터로 fit!
# 이렇게 하면 테스트 데이터 정보가 훈련에 누출됩니다.
'''
        
        with open(os.path.join(save_dir, "correct_usage_example.py"), 'w', encoding='utf-8') as f:
            f.write(example_code)
            
        print("=" * 60)
        print("✅ Stage1 전처리 완료! (Data Leakage 방지)")
        print("=" * 60)
        print(f"📊 최종 결과:")
        print(f"  📦 Stage1 데이터: {all_windows.shape} (스케일링 안됨)")
        print(f"  📋 라벨 분포: {metadata['stage1_label_distribution']}")
        print(f"  👥 전체 피험자: {len(unique_subjects)}명")
        print(f"  💾 저장 위치: {save_dir}")
        print(f"  🛡️  Data Leakage 방지: ✅")
        print(f"  📝 올바른 사용법: correct_usage_example.py 참조")
        print("=" * 60)
        print("⚠️  중요: 스케일링은 모델 훈련 시 GroupKFold 내부에서 수행하세요!")
        print("=" * 60)
        
        return save_dir

if __name__ == "__main__":
    preprocessor = GaitDataPreprocessor()
    save_path = preprocessor.process_and_save()
    print(f"Data Leakage가 방지된 전처리 데이터가 저장되었습니다: {save_path}")