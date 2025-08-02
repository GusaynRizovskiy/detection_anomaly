# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtCore
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.callbacks import Callback

# Настройка логирования для рабочего потока
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PlotCallback(Callback):
    """
    Класс обратного вызова для отправки данных о loss на каждой эпохе.
    """

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None and val_loss is not None:
            self.signal.emit({'epoch': epoch, 'loss': loss, 'val_loss': val_loss})


class Worker(QtCore.QObject):
    """
    Класс-работник для выполнения ресурсоемких операций в отдельном потоке.
    Наследуется от QObject для использования сигналов.
    """

    learning_finished = QtCore.pyqtSignal(dict)
    testing_finished = QtCore.pyqtSignal(dict)
    update_status_signal = QtCore.pyqtSignal(str)
    # НОВЫЙ СИГНАЛ: Отправляет данные для обновления графиков во время обучения
    update_plot_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.autoencoder = None
        self.scaler = None
        self.is_learning_running = False
        self.is_testing_running = False
        self.epoch_data = {'loss': [], 'val_loss': []}

    def load_and_preprocess_data(self, file_path, scaler, fit_scaler=True):
        """Загрузка, очистка и нормализация данных с обработкой кодировки и разделителей."""
        if not file_path:
            raise ValueError("Путь к файлу не указан.")

        data = None
        delimiters = [';', ',', '\t']

        # Перебираем возможные разделители
        for delimiter in delimiters:
            try:
                data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                if data.shape[1] > 1:
                    self.update_status_signal.emit(
                        f"✅ Файл '{os.path.basename(file_path)}' успешно загружен с разделителем '{delimiter}'.")
                    break
                else:
                    self.update_status_signal.emit(
                        f"⚠️ Файл загружен, но обнаружен только один столбец с разделителем '{delimiter}'. Пробуем другой разделитель...")
                    data = None
            except Exception:
                try:  # Пробуем cp1251
                    data = pd.read_csv(file_path, delimiter=delimiter, encoding='cp1251')
                    if data.shape[1] > 1:
                        self.update_status_signal.emit(
                            f"✅ Файл '{os.path.basename(file_path)}' успешно загружен с разделителем '{delimiter}' и кодировкой cp1251.")
                        break
                    else:
                        data = None
                except Exception:
                    self.update_status_signal.emit(
                        f"❌ Ошибка при загрузке с разделителем '{delimiter}'. Пробуем другой...")
                    data = None

        if data is None or data.shape[1] <= 1:
            raise ValueError(
                "Не удалось загрузить файл, так как ни один из предполагаемых разделителей (';', ',', '\\t') не подошел или файл содержит только один столбец.")

        data.columns = data.columns.str.strip()
        self.update_status_signal.emit(f"📝 Столбцы в файле: {', '.join(data.columns)}")

        if data.empty:
            raise ValueError("Загруженный CSV-файл пуст или не содержит данных.")

        if 'Время захвата пакетов' in data.columns:
            data.drop(columns=['Время захвата пакетов'], inplace=True)
            self.update_status_signal.emit("🗑️ Столбец 'Время захвата пакетов' удален.")

        data = data.replace(',', '.', regex=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        self.update_status_signal.emit(f"🔢 Преобразовано {data.shape[1]} столбцов в числовой формат.")

        rows_before = len(data)
        data = data.dropna()
        rows_after = len(data)
        if rows_before - rows_after > 0:
            self.update_status_signal.emit(f"❌ Удалено {rows_before - rows_after} строк с некорректными данными.")

        if data.empty:
            raise ValueError("В DataFrame не осталось данных после очистки. Проверьте формат данных в файле.")

        if fit_scaler or scaler is None:
            new_scaler = MinMaxScaler()
            scaled_data = new_scaler.fit_transform(data)
            self.update_status_signal.emit("📊 Данные масштабированы новым MinMaxScaler.")
            return scaled_data, new_scaler
        else:
            scaled_data = scaler.transform(data)
            self.update_status_signal.emit("📊 Данные масштабированы существующим MinMaxScaler.")
            return scaled_data, scaler

    def create_dataset(self, data, time_step):
        X = []
        for i in range(len(data) - time_step):
            a = data[i:(i + time_step), :]
            X.append(a)
        return np.array(X)

    def build_cnn_lstm_autoencoder(self, input_shape):
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
        if self.is_learning_running:
            self.update_status_signal.emit("⚠️ Обучение уже запущено.")
            return

        self.is_learning_running = True
        self.update_status_signal.emit("▶️ Начинаем обучение модели...")
        self.epoch_data = {'loss': [], 'val_loss': []}

        try:
            scaled_data, self.scaler = self.load_and_preprocess_data(file_path, self.scaler, fit_scaler=True)
            X_train = self.create_dataset(scaled_data, time_step)

            if X_train.size == 0:
                raise ValueError("Недостаточно данных для создания временных окон. Уменьшите 'Временной шаг'.")

            X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])
            self.autoencoder = self.build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))

            plot_callback = PlotCallback(self.update_plot_signal)

            history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                           verbose=0, callbacks=[plot_callback])

            self.update_status_signal.emit("🧠 Модель обучена. Расчет порога аномалий...")

            reconstruction_errors = np.mean(np.power(X_train - self.autoencoder.predict(X_train, verbose=0), 2),
                                            axis=(1, 2))
            threshold = np.percentile(reconstruction_errors, 95)

            results = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'threshold': threshold
            }

            self.learning_finished.emit(results)
            self.update_status_signal.emit("✅ Обучение завершено!")
        except Exception as e:
            self.update_status_signal.emit(f"❌ Произошла ошибка во время обучения: {e}")
            logging.error("Ошибка в рабочем потоке во время обучения", exc_info=True)
        finally:
            self.is_learning_running = False

    @QtCore.pyqtSlot(str, int, float)
    def start_testing(self, file_path, time_step, threshold):
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

            if X_new.size == 0:
                raise ValueError("Недостаточно данных для создания временных окон. Уменьшите 'Временной шаг'.")

            X_new = X_new.reshape(X_new.shape[0], time_step, scaled_data.shape[1])

            self.update_status_signal.emit("🔍 Выполняется предсказание на тестовых данных...")
            reconstruction_errors = np.mean(np.power(X_new - self.autoencoder.predict(X_new, verbose=0), 2),
                                            axis=(1, 2))
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
        pass