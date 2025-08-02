# -*- coding: utf-8 -*-
from PyQt5 import QtCore
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

# Настройка логирования для рабочего потока
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Worker(QtCore.QObject):
    """
    Класс-работник для выполнения ресурсоемких операций в отдельном потоке.
    Наследуется от QObject для использования сигналов.
    """

    # Сигналы для общения с основным потоком GUI
    learning_finished = QtCore.pyqtSignal(dict)  # Отправляет результаты обучения
    testing_finished = QtCore.pyqtSignal(dict)  # Отправляет результаты тестирования
    update_status_signal = QtCore.pyqtSignal(str)  # Отправляет сообщения для статусной строки

    def __init__(self, parent=None):
        super().__init__(parent)
        self.autoencoder = None
        self.scaler = None
        self.is_learning_running = False
        self.is_testing_running = False

    def load_and_preprocess_data(self, file_path, scaler, fit_scaler=True):
        """Загрузка, очистка и нормализация данных."""
        if not file_path:
            raise ValueError("Путь к файлу не указан.")

        data = pd.read_csv(file_path, delimiter=';', encoding='cp1251')
        data.columns = data.columns.str.strip()

        if 'Время захвата пакетов' in data.columns:
            data.drop(columns=['Время захвата пакетов'], inplace=True)

        data = data.replace(',', '.', regex=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()

        if fit_scaler or scaler is None:
            new_scaler = MinMaxScaler()
            scaled_data = new_scaler.fit_transform(data)
            return scaled_data, new_scaler
        else:
            scaled_data = scaler.transform(data)
            return scaled_data, scaler

    def create_dataset(self, data, time_step):
        """Создание временных рядов."""
        X = []
        for i in range(len(data) - time_step):
            a = data[i:(i + time_step), :]
            X.append(a)
        return np.array(X)

    def build_cnn_lstm_autoencoder(self, input_shape):
        """Создание архитектуры автоэнкодера."""
        input_layer = Input(shape=input_shape)
        conv_encoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
        lstm_encoder = LSTM(50, activation='relu', return_sequences=False)(conv_encoder)
        repeat = RepeatVector(input_shape[0])(lstm_encoder)
        lstm_decoder = LSTM(50, activation='relu', return_sequences=True)(repeat)
        conv_decoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(lstm_decoder)
        output_layer = Conv1D(input_shape[1], kernel_size=3, padding='same')(conv_decoder)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    @QtCore.pyqtSlot(str, int, int, int)
    def start_learning(self, file_path, time_step, epochs, batch_size):
        """Слот для запуска обучения модели."""
        if self.is_learning_running:
            self.update_status_signal.emit("⚠️ Обучение уже запущено.")
            return

        self.is_learning_running = True
        self.update_status_signal.emit("▶️ Начинаем обучение модели...")
        try:
            scaled_data, self.scaler = self.load_and_preprocess_data(file_path, self.scaler, fit_scaler=True)
            X_train = self.create_dataset(scaled_data, time_step)
            X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])
            self.autoencoder = self.build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))

            history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                           verbose=0)

            reconstruction_errors = np.mean(np.power(X_train - self.autoencoder.predict(X_train), 2), axis=(1, 2))
            threshold = np.percentile(reconstruction_errors, 95)

            results = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'threshold': threshold
            }

            self.learning_finished.emit(results)
            self.update_status_signal.emit("✅ Модель успешно обучена!")
        except Exception as e:
            self.update_status_signal.emit(f"❌ Произошла ошибка во время обучения: {e}")
            logging.error("Ошибка в рабочем потоке во время обучения", exc_info=True)
        finally:
            self.is_learning_running = False

    @QtCore.pyqtSlot(str, int, float)
    def start_testing(self, file_path, time_step, threshold):
        """Слот для запуска тестирования модели."""
        if self.is_testing_running:
            self.update_status_signal.emit("⚠️ Тестирование уже запущено.")
            return
        if self.autoencoder is None or self.scaler is None:
            self.update_status_signal.emit("⚠️ Ошибка: Модель не обучена или не загружена.")
            return

        self.is_testing_running = True
        self.update_status_signal.emit("▶️ Начинаем тестирование модели...")
        try:
            scaled_data, _ = self.load_and_preprocess_data(file_path, self.scaler, fit_scaler=False)
            X_new = self.create_dataset(scaled_data, time_step)
            X_new = X_new.reshape(X_new.shape[0], time_step, scaled_data.shape[1])

            reconstruction_errors = np.mean(np.power(X_new - self.autoencoder.predict(X_new), 2), axis=(1, 2))
            anomalies = np.where(reconstruction_errors > threshold)[0]

            results = {
                'reconstruction_errors': reconstruction_errors,
                'anomalies': anomalies
            }

            self.testing_finished.emit(results)
            self.update_status_signal.emit(f"✅ Тестирование завершено. Обнаружено {len(anomalies)} аномалий.")
        except Exception as e:
            self.update_status_signal.emit(f"❌ Произошла ошибка во время тестирования: {e}")
            logging.error("Ошибка в рабочем потоке во время тестирования", exc_info=True)
        finally:
            self.is_testing_running = False

    def stop(self):
        """Метод для остановки потока, если это необходимо."""
        # В этой реализации, поскольку операции не бесконечны, явная остановка не требуется,
        # но этот метод может быть полезен для более сложных задач.
        pass