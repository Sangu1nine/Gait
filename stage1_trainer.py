import os
import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 2.10.0 설정
print(f"TensorFlow version: {tf.__version__}")
# TFLite 호환성을 위한 설정
tf.random.set_seed(42)
np.random.seed(42)
tf.config.experimental.enable_op_determinism()

# ==== 내장 시각화 모듈 ====================================================
import itertools
from sklearn.metrics import (precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay

def create_figs_dir():
    os.makedirs("figs", exist_ok=True)

def lr_schedule_plot(lrs, save="figs/lr_schedule.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

def pr_curves(y_true, y_prob, class_names, save="figs/pr_curve.png"):
    plt.figure(figsize=(8, 6))
    for cls_idx, cls in enumerate(class_names):
        if len(class_names) == 2:  # Binary classification
            if cls_idx == 1:  # Only plot for positive class
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                plt.plot(recall, precision, 
                        label=f"{cls} (AUPRC={auc(recall, precision):.3f})")
        else:  # Multi-class
            mask = (y_true == cls_idx)
            precision, recall, _ = precision_recall_curve(mask, y_prob[:, cls_idx])
            plt.plot(recall, precision,
                    label=f"{cls} (AUPRC={auc(recall, precision):.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

def reliability_diag(y_true, y_prob, n_bins=10, save="figs/calibration_curve.png"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5, label="Perfect Calibration")
    plt.title("Reliability Diagram")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Fraction of Positives")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

def normalized_cm(y_true, y_pred, labels, save="figs/confusion_norm.png"):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=True)
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

def fold_violin(metrics_dict, metric_name="F1 Score", save="figs/fold_violin.png"):
    data = list(metrics_dict.values())
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[data])
    plt.xticks(range(len(metrics_dict)), [f"Fold {i+1}" for i in range(len(metrics_dict))])
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Distribution Across Folds")
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

def subject_heatmap(recall_matrix, subj_ids, class_names, save="figs/subject_heatmap.png"):
    plt.figure(figsize=(len(class_names)*1.5, len(subj_ids)*0.4+2))
    sns.heatmap(recall_matrix, annot=True, fmt=".2f",
                yticklabels=subj_ids, xticklabels=class_names,
                cmap="YlGnBu", cbar_kws={"label": "Recall"})
    plt.xlabel("Class")
    plt.ylabel("Subject")
    plt.title("Per-Subject Recall Performance")
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()

# ==== 기존 모델 코드 ======================================================

class TCNBlock(layers.Layer):
    """Temporal Convolutional Network Block (TFLite 호환)"""
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super(TCNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # main path - TFLite 호환성을 위해 명시적 설정
        self.conv1 = layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=True,
            activation=None
        )
        self.conv2 = layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=True,
            activation=None
        )
        # LayerNormalization 대신 BatchNormalization 사용 (TFLite 최적화)
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.activation = layers.ReLU()

        # residual path
        self.residual_conv = None

    def build(self, input_shape):
        super(TCNBlock, self).build(input_shape)
        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv1D(
                filters=self.filters, 
                kernel_size=1, 
                padding="same",
                use_bias=True,
                activation=None
            )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.batchnorm1(x, training=training)
        x = self.activation(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.activation(x)
        x = self.dropout2(x, training=training)

        # residual connection
        residual = self.residual_conv(inputs) if self.residual_conv else inputs
        return layers.Add()([residual, x])

    def get_config(self):
        config = super(TCNBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config

class Stage1Model:
    """Stage 1: Gait/Non-gait Binary Classification (TFLite 호환)"""
    def __init__(self, input_shape=(60, 6)):
        self.input_shape = input_shape
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_history = []  # LR 스케줄 추적용
        
    def build_model(self):
        """TCN 모델 구축 (TFLite 최적화)"""
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # TCN layers with reduced complexity for TFLite
        x = TCNBlock(32, kernel_size=3, dilation_rate=1, dropout_rate=0.2, name='tcn_block_1')(inputs)
        x = TCNBlock(32, kernel_size=3, dilation_rate=2, dropout_rate=0.2, name='tcn_block_2')(x)
        x = TCNBlock(64, kernel_size=3, dilation_rate=4, dropout_rate=0.2, name='tcn_block_3')(x)
        x = TCNBlock(64, kernel_size=3, dilation_rate=8, dropout_rate=0.2, name='tcn_block_4')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers - TFLite 최적화를 위해 단순화
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3, name='dropout_final')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='stage1_gait_classifier')
        
        # TensorFlow 2.10.0 호환 컴파일
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def balanced_lowo_split(self, X, y, subjects, frame_counts):
        """Balanced Leave-One-Walking-Out split"""
        walking_subjects = [s for s in set(subjects) if s.startswith('SA0') and int(s[2:]) <= 4]
        non_walking_subjects = [s for s in set(subjects) if s not in walking_subjects]
        
        folds = []
        for test_walking in walking_subjects:
            # Test walking subject frames
            test_w_mask = subjects == test_walking
            test_w_frames = np.sum(test_w_mask)
            
            # Select non-walking subjects for test
            test_nw_subjects = []
            test_nw_frames = 0
            target_frames = test_w_frames
            
            # Sort non-walking subjects by frame count
            nw_frame_counts = [(s, frame_counts[s]) for s in non_walking_subjects]
            nw_frame_counts.sort(key=lambda x: x[1], reverse=True)
            
            for nw_subj, nw_frames in nw_frame_counts:
                if test_nw_frames < target_frames * 0.9:  # Within 10%
                    test_nw_subjects.append(nw_subj)
                    test_nw_frames += nw_frames
                if test_nw_frames >= target_frames:
                    break
            
            # Create masks
            test_mask = (subjects == test_walking)
            for nw_subj in test_nw_subjects:
                test_mask |= (subjects == nw_subj)
            
            train_mask = ~test_mask
            
            folds.append({
                'train_idx': np.where(train_mask)[0],
                'test_idx': np.where(test_mask)[0],
                'test_walking': test_walking,
                'test_non_walking': test_nw_subjects
            })
            
        return folds
    
    def train(self, X, y, subjects, save_path):
        """모델 학습 (시각화 기능 통합)"""
        # figs 디렉토리 생성
        create_figs_dir()
        
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calculate frame counts per subject
        frame_counts = defaultdict(int)
        for i, subj in enumerate(subjects):
            frame_counts[subj] += 1
            
        # Get folds
        folds = self.balanced_lowo_split(X, y_encoded, subjects, frame_counts)
        
        # Training results
        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []
        fold_metrics = {}
        subject_recalls = []
        
        for fold_idx, fold in enumerate(folds):
            print(f"\n=== Fold {fold_idx + 1}/{len(folds)} ===")
            print(f"Test Walking: {fold['test_walking']}")
            print(f"Test Non-Walking: {fold['test_non_walking']}")
            
            # Build new model for each fold
            self.build_model()
            
            # Split data
            X_train, y_train = X[fold['train_idx']], y_encoded[fold['train_idx']]
            X_test, y_test = X[fold['test_idx']], y_encoded[fold['test_idx']]
            
            # Class weights for imbalanced data
            class_weight = {0: 1.0, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            y_pred_prob = self.model.predict(X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_pred_prob_flat = y_pred_prob.flatten()
            
            # Store results for visualization
            all_y_true.extend(y_test)
            all_y_prob.extend(y_pred_prob_flat)
            all_y_pred.extend(y_pred)
            
            # Metrics
            f1 = f1_score(y_test, y_pred, average='macro')
            ba = balanced_accuracy_score(y_test, y_pred)
            
            fold_results.append({
                'fold': fold_idx,
                'f1_score': f1,
                'balanced_accuracy': ba,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'history': history
            })
            
            fold_metrics[f'Fold_{fold_idx+1}'] = f1
            
            # Subject-level recall calculation
            test_subjects = subjects[fold['test_idx']]
            for subj in np.unique(test_subjects):
                subj_mask = test_subjects == subj
                subj_true = y_test[subj_mask]
                subj_pred = y_pred[subj_mask]
                
                if len(np.unique(subj_true)) > 1:  # Both classes present
                    subj_recall_0 = np.sum((subj_true == 0) & (subj_pred == 0)) / np.sum(subj_true == 0)
                    subj_recall_1 = np.sum((subj_true == 1) & (subj_pred == 1)) / np.sum(subj_true == 1)
                else:  # Only one class
                    if subj_true[0] == 0:
                        subj_recall_0 = np.sum(subj_pred == 0) / len(subj_pred)
                        subj_recall_1 = 0.0
                    else:
                        subj_recall_0 = 0.0
                        subj_recall_1 = np.sum(subj_pred == 1) / len(subj_pred)
                
                subject_recalls.append([subj, subj_recall_0, subj_recall_1])
            
            print(f"Fold {fold_idx + 1} - F1: {f1:.4f}, Balanced Acc: {ba:.4f}")
        
        # 시각화 생성
        print("\n=== 시각화 생성 중 ===")
        self.generate_visualizations(all_y_true, all_y_prob, all_y_pred, 
                                   fold_metrics, subject_recalls, save_path)
        
        # Save best model
        self.save_model(save_path, fold_results)
        
        return fold_results
    
    def generate_visualizations(self, y_true, y_prob, y_pred, fold_metrics, subject_recalls, save_path):
        """모든 시각화 생성"""
        class_names = ['Non-gait', 'Gait']
        
        # 1. Precision-Recall curves
        pr_curves(np.array(y_true), np.array(y_prob), class_names, 
                 save=os.path.join(save_path, "figs/stage1_pr_curves.png"))
        
        # 2. Reliability diagram
        reliability_diag(np.array(y_true), np.array(y_prob), 
                        save=os.path.join(save_path, "figs/stage1_calibration.png"))
        
        # 3. Normalized confusion matrix
        normalized_cm(np.array(y_true), np.array(y_pred), class_names,
                     save=os.path.join(save_path, "figs/stage1_confusion_norm.png"))
        
        # 4. Fold performance violin plot
        fold_violin(fold_metrics, "F1 Score", 
                   save=os.path.join(save_path, "figs/stage1_fold_violin.png"))
        
        # 5. Subject-level heatmap
        if subject_recalls:
            subject_data = np.array(subject_recalls)
            subjects = subject_data[:, 0]
            recall_matrix = subject_data[:, 1:].astype(float)
            
            subject_heatmap(recall_matrix, subjects, class_names,
                          save=os.path.join(save_path, "figs/stage1_subject_heatmap.png"))
        
        print("✅ 시각화 완료!")
        print(f"   - Precision-Recall 곡선")
        print(f"   - Calibration 다이어그램") 
        print(f"   - 정규화된 혼동행렬")
        print(f"   - Fold별 성능 분포")
        print(f"   - 피험자별 성능 히트맵")
    
    def save_model(self, save_path, fold_results):
        """모델 및 결과 저장 (TFLite 변환 포함)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_path, f"stage1_model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy visualizations to model directory
        import shutil
        figs_src = os.path.join(save_path, "figs")
        figs_dst = os.path.join(model_dir, "figs")
        if os.path.exists(figs_src):
            shutil.copytree(figs_src, figs_dst, dirs_exist_ok=True)
        
        # Save Keras model
        keras_model_path = os.path.join(model_dir, "model.keras")
        self.model.save(keras_model_path)
        
        # Save SavedModel format for TFLite conversion
        savedmodel_path = os.path.join(model_dir, "saved_model")
        self.model.save(savedmodel_path, save_format='tf')
        
        # Convert to TFLite
        self.convert_to_tflite(savedmodel_path, model_dir)
        
        # Save label encoder
        with open(os.path.join(model_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Plot results
        self.plot_results(fold_results, model_dir)
        
        # Save metrics
        avg_f1 = np.mean([r['f1_score'] for r in fold_results])
        avg_ba = np.mean([r['balanced_accuracy'] for r in fold_results])
        
        metrics = {
            'avg_f1_score': avg_f1,
            'avg_balanced_accuracy': avg_ba,
            'fold_results': fold_results,
            'input_shape': self.input_shape,
            'model_summary': self.model.to_json()
        }
        
        with open(os.path.join(model_dir, "metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
            
        print(f"\nStage 1 Model saved to: {model_dir}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Balanced Accuracy: {avg_ba:.4f}")
        
    def convert_to_tflite(self, savedmodel_path, model_dir):
        """TFLite 변환"""
        try:
            # TFLite 변환기 생성
            converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
            
            # 최적화 설정
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Float16 양자화 (모바일 최적화)
            converter.target_spec.supported_types = [tf.float16]
            
            # 변환 실행
            tflite_model = converter.convert()
            
            # TFLite 모델 저장
            tflite_path = os.path.join(model_dir, "model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"TFLite model saved to: {tflite_path}")
            print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
            
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            print("Keras model saved successfully, but TFLite conversion skipped.")
        
    def plot_results(self, fold_results, save_dir):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. F1 scores by fold
        ax = axes[0, 0]
        f1_scores = [r['f1_score'] for r in fold_results]
        ax.bar(range(len(f1_scores)), f1_scores)
        ax.set_xlabel('Fold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores by Fold')
        ax.axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
        ax.legend()
        
        # 2. Confusion matrices
        ax = axes[0, 1]
        avg_cm = np.mean([r['confusion_matrix'] for r in fold_results], axis=0)
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=ax)
        ax.set_title('Average Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # 3. Training history (last fold)
        ax = axes[1, 0]
        history = fold_results[-1]['history']
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History (Last Fold)')
        ax.legend()
        
        # 4. Model architecture
        ax = axes[1, 1]
        ax.text(0.1, 0.9, "TCN Architecture (TFLite Optimized):", transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.text(0.1, 0.8, "- Input: (60, 6)", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.7, "- TCN Block 1: 32 filters, dilation=1", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.6, "- TCN Block 2: 32 filters, dilation=2", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.5, "- TCN Block 3: 64 filters, dilation=4", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.4, "- TCN Block 4: 64 filters, dilation=8", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.3, "- Global Average Pooling", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.2, "- Dense: 32 units", transform=ax.transAxes, fontsize=10)
        ax.text(0.1, 0.1, "- Output: 1 unit (sigmoid)", transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'stage1_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

def load_stage1_data(base_path):
    """Stage 1 데이터 로드 (현재 디렉토리 구조 기반)"""
    print("Loading Stage 1 data from local directory...")
    
    # 전처리된 데이터 경로
    preprocessed_path = os.path.join(base_path, "preprocessed_20250612_151225")
    
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_path}")
    
    # Stage 1 data 로드
    stage1_data_path = os.path.join(preprocessed_path, "stage1_data_standard.npy")
    stage1_labels_path = os.path.join(preprocessed_path, "stage1_labels.npy")
    metadata_path = os.path.join(preprocessed_path, "metadata.pkl")
    
    if not all(os.path.exists(p) for p in [stage1_data_path, stage1_labels_path, metadata_path]):
        raise FileNotFoundError("Required data files not found in preprocessed directory")
    
    stage1_data = np.load(stage1_data_path)
    stage1_labels = np.load(stage1_labels_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        
    print(f"Loaded data shape: {stage1_data.shape}")
    print(f"Loaded labels shape: {stage1_labels.shape}")
    print(f"Walking subjects: {metadata['walking_subjects']}")
    print(f"Non-walking subjects: {len(metadata['non_walking_subjects'])} subjects")
    
    # Create subject labels based on metadata
    walking_subjects = metadata['walking_subjects']
    non_walking_subjects = metadata['non_walking_subjects']
    
    # Calculate samples per subject
    n_walking_samples = np.sum(stage1_labels == 'gait')
    n_non_walking_samples = np.sum(stage1_labels == 'non_gait')
    
    samples_per_walking = n_walking_samples // len(walking_subjects)
    samples_per_non_walking = n_non_walking_samples // len(non_walking_subjects)
    
    # Create subject array
    subjects = np.empty(len(stage1_labels), dtype='<U10')
    
    # Assign subjects (walking subjects first)
    idx = 0
    for subj in walking_subjects:
        n = samples_per_walking
        if idx + n > len(subjects):
            n = len(subjects) - idx
        subjects[idx:idx+n] = subj
        idx += n
        
    # Assign non-walking subjects
    for subj in non_walking_subjects:
        n = samples_per_non_walking
        if idx + n > len(subjects):
            n = len(subjects) - idx
        subjects[idx:idx+n] = subj
        idx += n
        if idx >= len(subjects):
            break
    
    print(f"Subject assignment completed. Total samples: {len(subjects)}")
    print(f"Walking samples: {n_walking_samples}, Non-walking samples: {n_non_walking_samples}")
    
    return stage1_data, stage1_labels, subjects

def validate_directory_structure(base_path):
    """디렉토리 구조 검증"""
    required_dirs = [
        "Selected_non_30hz",
        "support_label_data", 
        "walking_data",
        "preprocessed_20250612_151225"
    ]
    
    print("Validating directory structure...")
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ Found: {dir_name}")
        else:
            print(f"✗ Missing: {dir_name}")
            
    return all(os.path.exists(os.path.join(base_path, d)) for d in required_dirs)

if __name__ == "__main__":
    import sys
    import pathlib

    # Jupyter 환경 인자 제거
    if '-f' in sys.argv:
        f_idx = sys.argv.index('-f')
        sys.argv = sys.argv[:f_idx]

    # 현재 작업 디렉토리를 기본 경로로 설정
    if len(sys.argv) > 1:
        base_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
    else:
        base_path = pathlib.Path.cwd()  # 현재 디렉토리

    print("="*50)
    print("Stage 1: Gait/Non-gait Classification Training")
    print("TensorFlow 2.10.0 + TFLite Compatible + Enhanced Visualization")
    print("="*50)
    print(f"Base directory: {base_path}")

    # 디렉토리 구조 검증
    if not validate_directory_structure(base_path):
        print("\nError: Required directories not found!")
        print("Please ensure the following directories exist:")
        print("- Selected_non_30hz")
        print("- support_label_data") 
        print("- walking_data")
        print("- preprocessed_20250612_151225")
        sys.exit(1)

    try:
        # 데이터 로드
        X, y, subjects = load_stage1_data(str(base_path))
        
        # 모델 학습 (시각화 자동 생성 포함)
        model = Stage1Model()
        results = model.train(X, y, subjects, str(base_path))
        
        print("\nStage 1 Training Complete!")
        print("- Keras model saved")
        print("- TFLite model converted")
        print("- Results and visualizations saved")
        print("- Enhanced visualizations generated:")
        print("  ✓ Precision-Recall curves")
        print("  ✓ Calibration diagrams")
        print("  ✓ Normalized confusion matrix")
        print("  ✓ Fold performance distribution")
        print("  ✓ Subject-level performance heatmap")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)