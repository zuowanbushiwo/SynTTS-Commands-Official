import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import matplotlib

matplotlib.use('Agg')  # Optimize for headless servers
import matplotlib.pyplot as plt
import librosa
import re
from pathlib import Path
from collections import Counter
import argparse
import sys

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class EfficientNetB0Builder:
    """EfficientNet-B0 Builder adapted for Audio Spectrograms."""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_efficientnet_b0(self, pretrained=True, trainable_backbone=True):
        """
        Builds EfficientNet-B0 model.

        Args:
            pretrained: Whether to load ImageNet weights.
            trainable_backbone: Whether to fine-tune the backbone.
        """
        try:
            if pretrained:
                # Load ImageNet weights (requires 3-channel input)
                # We define input shape as (H, W, 3) for the backbone
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(self.input_shape[0], self.input_shape[1], 3)
                )

                # Adapter: 1-Channel Audio -> 3-Channel Image format
                inputs = tf.keras.Input(shape=self.input_shape)
                # 1x1 Convolution to expand 1 channel to 3 channels
                x = layers.Conv2D(3, (1, 1), padding='same', name='expand_channels')(inputs)
            else:
                # Train from scratch (can accept 1 channel if we config it, but keeping adapter is safer for arch consistency)
                base_model = EfficientNetB0(
                    weights=None,
                    include_top=False,
                    input_shape=(self.input_shape[0], self.input_shape[1], 3)
                )
                inputs = tf.keras.Input(shape=self.input_shape)
                x = layers.Conv2D(3, (1, 1), padding='same', name='expand_channels')(inputs)

            # Freeze/Unfreeze backbone
            base_model.trainable = trainable_backbone

            # Fine-tuning strategy: Unfreeze top layers if requested
            if trainable_backbone and pretrained:
                # Unfreeze last 30 layers
                for layer in base_model.layers[:-30]:
                    layer.trainable = False
                for layer in base_model.layers[-30:]:
                    layer.trainable = True

            # Forward pass through backbone
            x = base_model(x, training=trainable_backbone)

            # Classification Head (Adapted for KWS)
            x = layers.GlobalAveragePooling2D(name='global_pool')(x)
            x = layers.Dropout(0.3, name='dropout1')(x)
            x = layers.Dense(512, activation='relu', name='fc1')(x)
            x = layers.Dropout(0.3, name='dropout2')(x)
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

            model = models.Model(inputs, outputs, name='efficientnet_b0_kws')
            return model

        except ImportError:
            print("[Warning] EfficientNetB0 not available, using custom implementation.")
            return self._build_alternative_efficientnet()

    def _build_alternative_efficientnet(self):
        """Simplified EfficientNet-like fallback."""
        print("[Info] Building simplified EfficientNet-B0...")
        # (Simplified implementation omitted for brevity, assuming TensorFlow is installed correctly)
        # Using a standard CNN as fallback if Import fails
        inputs = tf.keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs)


class UnifiedAudioDataLoader:
    """Robust Audio Loader handling both Chinese and English subsets."""

    def __init__(self, base_dir, sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

    def normalize_path(self, path):
        return str(Path(path).as_posix())

    def _extract_label(self, file_path):
        """Extracts label from folder structure: Subset/Label/file.wav"""
        file_path = self.normalize_path(file_path)
        parts = file_path.split('/')

        if len(parts) < 2: return "unknown"

        # The label is typically the folder name (index 1 if path is relative like 'Free_ST_Chinese/Label/...')
        # Adjust based on your list file format (assuming relative path from root)
        folder_name = parts[1]

        # Handle SynTTS specific naming quirks
        wake_word_mapping = {
            'Hello_小智': 'Hello 小智',
            '嗨_三星小贝': '嗨 三星小贝',
            'Hello_XiaoZhi': 'Hello 小智',  # Add potential variants if any
        }

        if folder_name in wake_word_mapping:
            return wake_word_mapping[folder_name]

        # General cleaning
        # Remove file extensions if folder name has dots
        if '.' in folder_name:
            folder_name = folder_name.split('.')[0]

        return folder_name

    def create_label_mapping(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"[Info] Label mapping created: {len(unique_labels)} classes")
        return self.label_to_idx

    def preload_data(self, list_paths):
        print("[Info] Starting data preloading...")
        all_paths = []
        all_labels = []

        for p in list_paths:
            if not os.path.exists(p):
                print(f"[Warning] List not found: {p}")
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    rel_path = line.strip()
                    full_path = os.path.join(self.base_dir, rel_path)
                    if os.path.exists(full_path):
                        all_paths.append(full_path)
                        all_labels.append(self._extract_label(rel_path))

        self.create_label_mapping(all_labels)

        spectrograms = []
        indices = []
        valid_paths = []

        print(f"[Info] Loading {len(all_paths)} files...")
        for i, path in enumerate(all_paths):
            if i % 2000 == 0: print(f"       Progress: {i}/{len(all_paths)}")
            try:
                # Load
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
                # Spec
                spec = self._audio_to_mel(audio)

                spectrograms.append(spec)
                indices.append(self.label_to_idx[all_labels[i]])
                valid_paths.append(path)  # Store full path
            except Exception as e:
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(indices)
        self.preloaded_data['paths'] = valid_paths  # For mapping back to splits

        print(f"[Info] Preloading done. Valid samples: {len(spectrograms)}")
        return self.preloaded_data

    def create_dataset(self, split_list, batch_size):
        # 1. Load split list paths
        target_paths = set()
        with open(split_list, 'r', encoding='utf-8') as f:
            for line in f:
                target_paths.add(self.normalize_path(os.path.join(self.base_dir, line.strip())))

        # 2. Filter preloaded data
        indices = []
        # Optimization: Create map for O(1) lookup
        # (Assuming paths in preloaded_data match normalization of target_paths)
        loaded_path_map = {self.normalize_path(p): i for i, p in enumerate(self.preloaded_data['paths'])}

        for tp in target_paths:
            if tp in loaded_path_map:
                indices.append(loaded_path_map[tp])

        print(f"[Info] Creating dataset from {os.path.basename(split_list)}: {len(indices)} samples found")

        specs = self.preloaded_data['spectrograms'][indices]
        lbls = self.preloaded_data['labels'][indices]

        ds = tf.data.Dataset.from_tensor_slices((specs, lbls))
        ds = ds.shuffle(len(indices)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _audio_to_mel(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=40, hop_length=256, win_length=1024, fmin=20, fmax=4000
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad/Crop to 49
        target = 49
        if log_mel.shape[1] < target:
            pad = target - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
        else:
            log_mel = log_mel[:, :target]

        return np.expand_dims(log_mel.T, -1).astype(np.float32)


class ModelTrainer:
    def __init__(self, model, output_dir, model_name):
        self.model = model
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = None
        os.makedirs(output_dir, exist_ok=True)

    def compile(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs):
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, f'best_{self.model_name}.keras'),
                save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
        ]
        self.history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=cbs, verbose=1)
        self._plot_history()

    def evaluate(self, test_ds):
        print("[Info] Evaluating...")
        res = self.model.evaluate(test_ds, verbose=1)
        return dict(zip(self.model.metrics_names, res))

    def _plot_history(self):
        if not self.history: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy');
        ax1.legend()
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set_title('Loss');
        ax2.legend()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_history.png'))
        plt.close()

    def save_report(self, res, args):
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump({'config': vars(args), 'results': res}, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on SynTTS")

    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language')
    parser.add_argument('--output_dir', type=str, default='./results/efficientnet')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'])

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--pretrained', action='store_true', default=True, help="Use ImageNet weights")

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Config: EfficientNet-B0 on {args.language} ---")
    print(vars(args))

    # 1. Loader
    loader = UnifiedAudioDataLoader(args.dataset_root, args.sample_rate)

    train_list = os.path.join(args.splits_dir, f'train_list_{args.language}.txt')
    val_list = os.path.join(args.splits_dir, f'validation_list_{args.language}.txt')
    test_list = os.path.join(args.splits_dir, f'test_list_{args.language}.txt')

    loader.preload_data([train_list, val_list, test_list])

    # 2. Datasets
    train_ds = loader.create_dataset(train_list, args.batch_size)
    val_ds = loader.create_dataset(val_list, args.batch_size)
    test_ds = loader.create_dataset(test_list, args.batch_size)

    # 3. Model
    # Input shape: (Time, Mels, 1) -> (49, 40, 1)
    builder = EfficientNetB0Builder((49, 40, 1), len(loader.label_to_idx))
    model = builder.build_efficientnet_b0(pretrained=args.pretrained, trainable_backbone=True)
    model.summary()

    # 4. Train
    trainer = ModelTrainer(model, args.output_dir, f'effnet_{args.language}')
    trainer.compile(args.lr)
    trainer.train(train_ds, val_ds, args.epochs)

    # 5. Eval
    res = trainer.evaluate(test_ds)
    print(f"\n[Result] Test Acc: {res['accuracy']:.4f}")
    trainer.save_report(res, args)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    main()