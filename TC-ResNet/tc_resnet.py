import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
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


class TCResNetBuilder:
    """
    TC-ResNet Model Builder (Temporal Convolutional ResNet).
    Paper: "Temporal Convolution for Real-time Keyword Spotting on Mobile Devices"
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _temporal_conv_block(self, x, filters, kernel_size, stride=1, dilation_rate=1, name=""):
        """Temporal Convolution Block with Residual Connection"""
        shortcut = x

        # First Conv (Time dimension only: kernel_size x 1)
        x = layers.Conv2D(filters, (kernel_size, 1),
                          strides=(stride, 1),
                          padding='same',
                          dilation_rate=(dilation_rate, 1),
                          name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.ReLU(name=f'{name}_relu1')(x)
        x = layers.Dropout(0.2, name=f'{name}_dropout1')(x)

        # Second Conv
        x = layers.Conv2D(filters, (kernel_size, 1),
                          strides=1,
                          padding='same',
                          dilation_rate=(dilation_rate, 1),
                          name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)

        # Shortcut handling (if dimensions change)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1),
                                     strides=(stride, 1),
                                     padding='same',
                                     name=f'{name}_shortcut_conv')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

        # Add & ReLU
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name}_out_relu')(x)

        return x

    def build_tc_resnet(self, model_size='small'):
        """
        Builds TC-ResNet variants.
        Args:
            model_size: 'small', 'medium', 'large'
        """
        inputs = tf.keras.Input(shape=self.input_shape)

        # Configuration Dictionaries
        configs = {
            'small': {
                'filters': [16, 24, 32, 48],
                'kernel_sizes': [9, 9, 9, 9],
                'strides': [1, 2, 2, 2],
                'units': 64
            },
            'medium': {
                'filters': [32, 48, 64, 96],
                'kernel_sizes': [9, 9, 9, 9],
                'strides': [1, 2, 2, 2],
                'units': 128
            },
            'large': {
                'filters': [64, 96, 128, 192],
                'kernel_sizes': [9, 9, 9, 9],
                'strides': [1, 2, 2, 2],
                'units': 256
            }
        }

        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")

        cfg = configs[model_size]

        # Stem Conv
        x = layers.Conv2D(cfg['filters'][0], (3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='stem_conv')(inputs)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.ReLU(name='stem_relu')(x)
        x = layers.Dropout(0.2, name='stem_dropout')(x)

        # TC-Blocks
        for i, (filters, kernel_size, stride) in enumerate(zip(
                cfg['filters'][1:], cfg['kernel_sizes'][1:], cfg['strides'][1:])):
            x = self._temporal_conv_block(
                x, filters, kernel_size, stride=stride, name=f'block{i + 1}')

        # Classification Head
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(cfg['units'], activation='relu', name='fc1')(x)
        x = layers.Dropout(0.3, name='dropout_fc')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        model = models.Model(inputs, outputs, name=f'tc_resnet_{model_size}')
        return model


class UnifiedAudioDataLoader:
    """
    Robust Data Loader handling both Chinese and English subsets.
    """

    def __init__(self, base_dir, sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

        # English Command List
        self.english_commands = [
            "Play", "Pause", "Resume", "Play from start", "Repeat song",
            "Previous track", "Next track", "Last song", "Skip song", "Jump to first track",
            "Volume up", "Volume down", "Mute", "Set volume to 50%", "Max volume",
            "Answer call", "Hang up", "Decline call",
            "Hey Siri", "OK Google", "Hey Google", "Alexa", "Hi Bixby"
        ]

    def normalize_path(self, path):
        return str(Path(path).as_posix())

    def _contains_chinese(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _extract_label(self, file_path):
        """Unified label extraction logic."""
        file_path = self.normalize_path(file_path)
        parts = file_path.split('/')

        if len(parts) < 2: return "unknown"

        folder_name = parts[1]

        # 1. Wake Words (Mixed)
        wake_mapping = {
            'Hello_小智': 'Hello 小智',
            '嗨_三星小贝': '嗨 三星小贝',
        }
        if folder_name in wake_mapping: return wake_mapping[folder_name]

        # 2. Chinese Logic
        if '_' in folder_name:
            folder_parts = folder_name.split('_')
            if self._contains_chinese(folder_parts[-1]):
                return folder_parts[-1]

        if self._contains_chinese(folder_name):
            if '.' in folder_name: return folder_name.split('.')[0]
            return folder_name

        # 3. English Logic
        path_lower = file_path.lower()
        for cmd in self.english_commands:
            # Match "set_volume_to_50%" format
            cmd_slug = cmd.lower().replace(' ', '_').replace('50%', '50')
            if cmd_slug in path_lower:
                return cmd

        # Fallback
        if '.' in folder_name:
            return folder_name.split('.')[0]

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
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
                spec = self._audio_to_mel(audio)
                spectrograms.append(spec)
                indices.append(self.label_to_idx[all_labels[i]])
                valid_paths.append(self.normalize_path(path))
            except Exception as e:
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(indices)
        self.preloaded_data['paths'] = valid_paths

        print(f"[Info] Preloading done. Valid samples: {len(spectrograms)}")
        return self.preloaded_data

    def create_dataset(self, split_list, batch_size):
        target_paths = set()
        with open(split_list, 'r', encoding='utf-8') as f:
            for line in f:
                target_paths.add(self.normalize_path(os.path.join(self.base_dir, line.strip())))

        path_to_idx = {p: i for i, p in enumerate(self.preloaded_data['paths'])}

        indices = []
        for tp in target_paths:
            if tp in path_to_idx:
                indices.append(path_to_idx[tp])

        print(f"[Info] Creating dataset from {os.path.basename(split_list)}: {len(indices)} samples")

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
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
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
    parser = argparse.ArgumentParser(description="Train TC-ResNet on SynTTS")

    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language')
    parser.add_argument('--output_dir', type=str, default='./results/tcresnet')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'])

    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=int, default=16000)

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Config: TC-ResNet ({args.model_size}) on {args.language} ---")
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
    builder = TCResNetBuilder((49, 40, 1), len(loader.label_to_idx))
    model = builder.build_tc_resnet(args.model_size)
    model.summary()

    # 4. Train
    trainer = ModelTrainer(model, args.output_dir, f'tcresnet_{args.model_size}_{args.language}')
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