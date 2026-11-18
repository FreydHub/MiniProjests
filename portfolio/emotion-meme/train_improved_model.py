#!/usr/bin/env python3
"""
Скрипт для обучения улучшенной модели классификации эмоций
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from PIL import Image
import logging
from pathlib import Path
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedEmotionTrainer:
    """
    Класс для обучения улучшенной модели классификации эмоций
    """
    
    def __init__(self, data_path="data/datasets", model_path="data/models"):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Создание папок
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.model_path / 'checkpoints').mkdir(exist_ok=True)
        (self.model_path / 'logs').mkdir(exist_ok=True)
        (self.model_path / 'pretrained').mkdir(exist_ok=True)
        
        self.model = None
        self.history = None
    
    def create_advanced_model(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Создание продвинутой архитектуры модели
        """
        logger.info("Создание продвинутой архитектуры модели...")
        
        # Используем функциональный API для более сложной архитектуры
        inputs = tf.keras.Input(shape=input_shape)
        
        # Первый блок свертки
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Второй блок свертки
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Третий блок свертки
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Четвертый блок свертки
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Полносвязные слои
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Улучшенный оптимизатор
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Компиляция модели
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model
    
    def load_fer2013_data(self, use_images=True):
        """
        Загрузка данных FER2013
        Основано на подходе из https://github.com/atulapra/Emotion-detection
        
        Args:
            use_images: Если True, загружает изображения из папок. Если False, загружает из CSV.
        """
        logger.info("Загрузка данных FER2013...")
        
        # Попытка загрузки из подготовленных изображений
        if use_images:
            processed_dir = self.data_path / "processed"
            train_dir = processed_dir / "train"
            
            if train_dir.exists() and any(train_dir.iterdir()):
                logger.info("Загрузка данных из подготовленных изображений...")
                return self._load_from_images()
        
        # Попытка загрузки из CSV
        csv_path = self.data_path / "raw" / "fer2013.csv"
        if not csv_path.exists():
            csv_path = self.data_path / "raw" / "fer2013_sample.csv"
        
        if csv_path.exists():
            logger.info(f"Загрузка данных из CSV: {csv_path}")
            return self._load_from_csv(csv_path)
        
        logger.warning("Данные FER2013 не найдены, создаем синтетические данные")
        logger.info("Запустите: python download_datasets.py")
        logger.info("Затем: python dataset_prepare.py")
        return self._create_synthetic_data()
    
    def _load_from_images(self):
        """
        Загрузка данных из подготовленных изображений
        """
        processed_dir = self.data_path / "processed"
        train_dir = processed_dir / "train"
        
        X = []
        y = []
        
        # Загрузка обучающих данных
        for emotion_idx, emotion_name in enumerate(self.emotion_labels):
            emotion_dir = train_dir / emotion_name
            if not emotion_dir.exists():
                continue
            
            image_files = list(emotion_dir.glob("*.png")) + list(emotion_dir.glob("*.jpg"))
            
            for img_path in image_files:
                try:
                    # Загрузка изображения
                    img = Image.open(img_path).convert('L')  # Grayscale
                    img_array = np.array(img, dtype='float32')
                    
                    # Нормализация
                    img_array = img_array / 255.0
                    
                    # Изменение размера до 48x48 если нужно
                    if img_array.shape != (48, 48):
                        img = img.resize((48, 48))
                        img_array = np.array(img, dtype='float32') / 255.0
                    
                    X.append(img_array)
                    y.append(emotion_idx)
                    
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {img_path}: {e}")
                    continue
        
        if len(X) == 0:
            logger.warning("Не найдено изображений, используем CSV или синтетические данные")
            return self._load_from_csv() if (self.data_path / "raw" / "fer2013.csv").exists() else self._create_synthetic_data()
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, len(self.emotion_labels))
        
        logger.info(f"Загружено {len(X)} изображений из папок")
        return X, y
    
    def _load_from_csv(self, csv_path=None):
        """
        Загрузка данных из CSV файла
        """
        if csv_path is None:
            csv_path = self.data_path / "raw" / "fer2013.csv"
            if not csv_path.exists():
                csv_path = self.data_path / "raw" / "fer2013_sample.csv"
        
        if not csv_path.exists():
            return self._create_synthetic_data()
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Загружено {len(df)} записей из CSV")
            
            # Обработка данных
            X = []
            y = []
            
            for _, row in df.iterrows():
                try:
                    # Парсинг пикселей
                    pixels = np.array(row['pixels'].split(), dtype='uint8')
                    pixels = pixels.reshape(48, 48)
                    
                    # Нормализация
                    pixels = pixels.astype('float32') / 255.0
                    
                    X.append(pixels)
                    y.append(int(row['emotion']))
                except Exception as e:
                    logger.warning(f"Ошибка обработки строки: {e}")
                    continue
            
            X = np.array(X)
            y = tf.keras.utils.to_categorical(y, len(self.emotion_labels))
            
            logger.info(f"Обработано {len(X)} изображений из CSV")
            return X, y
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных из CSV: {e}")
            return self._create_synthetic_data()
    
    def load_data_with_splits(self):
        """
        Загрузка данных с разделением на train/validation/test из папок
        """
        processed_dir = self.data_path / "processed"
        
        splits = {
            'train': processed_dir / "train",
            'validation': processed_dir / "validation",
            'test': processed_dir / "test"
        }
        
        data_splits = {}
        
        for split_name, split_dir in splits.items():
            if not split_dir.exists():
                logger.warning(f"Папка {split_name} не найдена")
                continue
            
            X = []
            y = []
            
            for emotion_idx, emotion_name in enumerate(self.emotion_labels):
                emotion_dir = split_dir / emotion_name
                if not emotion_dir.exists():
                    continue
                
                image_files = list(emotion_dir.glob("*.png")) + list(emotion_dir.glob("*.jpg"))
                
                for img_path in image_files:
                    try:
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img, dtype='float32') / 255.0
                        
                        if img_array.shape != (48, 48):
                            img = img.resize((48, 48))
                            img_array = np.array(img, dtype='float32') / 255.0
                        
                        X.append(img_array)
                        y.append(emotion_idx)
                    except Exception as e:
                        logger.warning(f"Ошибка загрузки {img_path}: {e}")
                        continue
            
            if len(X) > 0:
                X = np.array(X)
                y = tf.keras.utils.to_categorical(y, len(self.emotion_labels))
                data_splits[split_name] = (X, y)
                logger.info(f"{split_name}: {len(X)} изображений")
        
        return data_splits
    
    def _create_synthetic_data(self, num_samples=5000):
        """
        Создание синтетических данных для демонстрации
        """
        logger.info("Создание синтетических данных...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            # Создание случайного изображения с паттернами
            img = np.random.rand(48, 48).astype('float32')
            
            # Добавление простых паттернов для разных эмоций
            emotion = i % len(self.emotion_labels)
            
            if emotion == 0:  # angry - темные цвета
                img = img * 0.3 + 0.1
            elif emotion == 1:  # disgust - зеленоватые тона
                img[:, :24] = img[:, :24] * 0.5 + 0.3
            elif emotion == 2:  # fear - синие тона
                img = img * 0.4 + 0.2
            elif emotion == 3:  # happy - светлые тона
                img = img * 0.8 + 0.2
            elif emotion == 4:  # sad - серые тона
                img = img * 0.4 + 0.1
            elif emotion == 5:  # surprise - контрастные
                img = np.where(img > 0.5, 0.9, 0.1)
            else:  # neutral - средние тона
                img = img * 0.5 + 0.25
            
            X.append(img)
            y.append(emotion)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, len(self.emotion_labels))
        
        return X, y
    
    def create_data_generators(self, X_train, X_val, y_train, y_val, batch_size=32):
        """
        Создание генераторов данных с аугментацией
        """
        logger.info("Создание генераторов данных...")
        
        # Аугментация для обучающих данных
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Только нормализация для валидационных данных
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Подготовка данных для генераторов
        X_train_expanded = np.expand_dims(X_train, axis=-1)
        X_val_expanded = np.expand_dims(X_val, axis=-1)
        
        train_generator = train_datagen.flow(X_train_expanded, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val_expanded, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def train_model(self, epochs=100, batch_size=32, use_prepared_splits=True):
        """
        Обучение модели
        Основано на подходе из https://github.com/atulapra/Emotion-detection
        
        Args:
            epochs: Количество эпох
            batch_size: Размер батча
            use_prepared_splits: Если True, использует предварительно разделенные данные из папок
        """
        logger.info("Начало обучения улучшенной модели...")
        logger.info("Основано на: https://github.com/atulapra/Emotion-detection")
        
        # Попытка загрузки данных с разделением из папок
        if use_prepared_splits:
            data_splits = self.load_data_with_splits()
            
            if 'train' in data_splits and 'validation' in data_splits:
                logger.info("Использование предварительно разделенных данных из папок")
                X_train, y_train = data_splits['train']
                X_val, y_val = data_splits['validation']
                
                if 'test' in data_splits:
                    X_test, y_test = data_splits['test']
                else:
                    X_test, y_test = X_val, y_val
            else:
                logger.info("Предварительно разделенные данные не найдены, используем CSV")
                X, y = self.load_fer2013_data(use_images=False)
                
                # Разделение данных
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
                )
        else:
            # Загрузка данных и разделение вручную
            X, y = self.load_fer2013_data(use_images=True)
            
            # Разделение данных
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
            )
        
        logger.info(f"Обучающих данных: {len(X_train)}")
        logger.info(f"Валидационных данных: {len(X_val)}")
        logger.info(f"Тестовых данных: {len(X_test)}")
        
        # Создание модели
        self.model = self.create_advanced_model()
        
        # Создание генераторов данных
        train_gen, val_gen = self.create_data_generators(
            X_train, X_val, y_train, y_val, batch_size
        )
        
        # Сохранение тестовых данных для оценки
        self.X_test = X_test
        self.y_test = y_test
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path / 'checkpoints' / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.model_path / 'logs'),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Обучение
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Сохранение модели
        model_save_path = self.model_path / 'pretrained' / 'emotion_model.h5'
        self.model.save(str(model_save_path))
        logger.info(f"Модель сохранена: {model_save_path}")
        
        # Оценка модели на тестовых данных
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            logger.info("Оценка модели на тестовых данных...")
            self.evaluate_model(self.X_test, self.y_test)
        else:
            logger.info("Оценка модели на валидационных данных...")
            self.evaluate_model(X_val, y_val)
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Оценка модели
        """
        logger.info("Оценка модели...")
        
        # Проверка и подготовка данных
        try:
            # Убеждаемся, что данные имеют правильную форму
            if len(X_test.shape) == 3:  # (samples, height, width)
                # Добавляем канальный размер
                X_test = np.expand_dims(X_test, axis=-1)
                logger.info(f"Данные расширены до формы: {X_test.shape}")
            
            # Проверяем форму данных
            logger.info(f"Форма X_test: {X_test.shape}")
            logger.info(f"Форма y_test: {y_test.shape}")
            logger.info(f"Ожидаемая форма входа модели: {self.model.input_shape}")
        except Exception as e:
            logger.warning(f"Ошибка при подготовке данных: {e}")
        
        # Предсказания
        try:
            y_pred = self.model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
            raise
        
        # Метрики - используем безопасный подход с перекомпиляцией при необходимости
        test_loss = 0.0
        test_accuracy = 0.0
        test_top3_accuracy = 0.0
        
        # Сохраняем оригинальную компиляцию
        original_optimizer = self.model.optimizer
        original_loss = self.model.loss
        original_metrics = self.model.metrics_names if hasattr(self.model, 'metrics_names') else []
        
        try:
            # Попытка оценки с оригинальными метриками
            evaluation_results = self.model.evaluate(X_test, y_test, verbose=0)
            if isinstance(evaluation_results, list):
                if len(evaluation_results) >= 1:
                    test_loss = float(evaluation_results[0])
                if len(evaluation_results) >= 2:
                    test_accuracy = float(evaluation_results[1])
                if len(evaluation_results) >= 3:
                    test_top3_accuracy = float(evaluation_results[2])
        except Exception as e:
            logger.warning(f"Ошибка при оценке модели с метриками: {e}")
            logger.info("Перекомпиляция модели с безопасными метриками...")
            
            # Перекомпилируем модель только с базовыми метриками
            try:
                self.model.compile(
                    optimizer=original_optimizer,
                    loss=original_loss,
                    metrics=['accuracy']  # Только accuracy, без top_3_accuracy
                )
                
                # Пробуем снова оценить
                evaluation_results = self.model.evaluate(X_test, y_test, verbose=0)
                if isinstance(evaluation_results, list):
                    if len(evaluation_results) >= 1:
                        test_loss = float(evaluation_results[0])
                    if len(evaluation_results) >= 2:
                        test_accuracy = float(evaluation_results[1])
            except Exception as e2:
                logger.warning(f"Ошибка при перекомпиляции: {e2}")
                logger.info("Вычисление метрик вручную...")
                
                # Вычисление loss вручную
                try:
                    from tensorflow.keras.losses import categorical_crossentropy
                    test_loss = float(np.mean(categorical_crossentropy(y_test, y_pred).numpy()))
                except Exception as e3:
                    logger.warning(f"Не удалось вычислить loss: {e3}")
                
                # Вычисление accuracy вручную
                test_accuracy = float(np.mean(y_pred_classes == y_true_classes))
        
        # Вычисление Top-3 accuracy вручную
        try:
            # Получаем топ-3 предсказаний
            top3_pred = np.argsort(y_pred, axis=1)[:, -3:]
            # Проверяем, есть ли правильный класс в топ-3
            top3_correct = np.array([y_true_classes[i] in top3_pred[i] for i in range(len(y_true_classes))])
            test_top3_accuracy = float(np.mean(top3_correct))
        except Exception as e:
            logger.warning(f"Не удалось вычислить top-3 accuracy: {e}")
            test_top3_accuracy = 0.0
        
        logger.info(f"Точность: {test_accuracy:.4f}")
        logger.info(f"Top-3 точность: {test_top3_accuracy:.4f}")
        logger.info(f"Loss: {test_loss:.4f}")
        
        # Отчет по классификации
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        # Сохранение метрик
        metrics = {
            'test_accuracy': float(test_accuracy),
            'test_top3_accuracy': float(test_top3_accuracy),
            'test_loss': float(test_loss),
            'classification_report': report
        }
        
        with open(str(self.model_path / 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        return metrics
    
    def plot_training_history(self):
        """
        Визуализация истории обучения
        """
        if self.history is None:
            logger.warning("История обучения не найдена")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График точности
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # График потерь
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # График Top-3 точности
        if 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3')
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3')
            axes[1, 0].set_title('Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # График learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(str(self.model_path / 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """
        Визуализация матрицы ошибок
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(str(self.model_path / 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Основная функция для обучения улучшенной модели
    """
    logger.info("🚀 Запуск обучения улучшенной модели классификации эмоций")
    
    # Создание тренера
    trainer = ImprovedEmotionTrainer()
    
    # Обучение модели
    history = trainer.train_model(epochs=50, batch_size=32)
    
    # Визуализация результатов
    trainer.plot_training_history()
    
    logger.info("🎉 Обучение завершено!")
    logger.info("📊 Результаты сохранены в data/models/")
    logger.info("🔧 Для использования улучшенной модели перезапустите API сервер")

if __name__ == "__main__":
    main()
