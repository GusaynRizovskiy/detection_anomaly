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
import socket
import json
import collections
from datetime import datetime

logger = logging.getLogger(__name__)


# Класс обратного вызова для отрисовки графиков обучения
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


# --- Класс для задач машинного обучения ---
class MLWorker(QtCore.QObject):
    """
    Класс-работник для выполнения ресурсоемких операций ML в отдельном потоке.
    """
    learning_finished = QtCore.pyqtSignal(dict)
    testing_finished = QtCore.pyqtSignal(dict)
    online_results_signal = QtCore.pyqtSignal(dict)
    update_status_signal = QtCore.pyqtSignal(str)
    update_plot_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.autoencoder = None
        self.scaler = None
        self.is_learning_running = False
        self.is_testing_running = False

    def load_and_preprocess_data(self, file_path, scaler, fit_scaler=True):
        """Загрузка, очистка и нормализация данных с обработкой кодировки и разделителей."""
        if not file_path:
            logger.error("Путь к файлу не указан.")
            raise ValueError("Путь к файлу не указан.")

        data = None
        delimiters = [';', ',', '\t']
        encodings = ['utf-8', 'cp1251']

        logger.info(f"Начинаем загрузку файла: {file_path}")
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                    if data.shape[1] > 1:
                        self.update_status_signal.emit(
                            f"✅ Файл '{os.path.basename(file_path)}' успешно загружен с разделителем '{delimiter}' и кодировкой '{encoding}'.")
                        logger.info(f"Файл успешно загружен. Разделитель: {delimiter}, Кодировка: {encoding}")
                        break
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить файл с разделителем '{delimiter}' и кодировкой '{encoding}': {e}")
                    data = None
            if data is not None and data.shape[1] > 1:
                break

        if data is None or data.shape[1] <= 1:
            logger.error("Не удалось загрузить файл или он не содержит достаточного количества столбцов.")
            raise ValueError(
                "Не удалось загрузить файл. Пожалуйста, убедитесь, что файл CSV не пуст, содержит более одного столбца и использует стандартные разделители (';', ',', '\\t').")

        data.columns = data.columns.str.strip()
        self.update_status_signal.emit(f"📝 Столбцы в файле: {', '.join(data.columns)}")

        if data.empty:
            logger.error("Загруженный CSV-файл пуст после загрузки.")
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
            logger.warning(f"Удалено {rows_before - rows_after} строк с некорректными данными.")

        if data.empty:
            logger.error("В DataFrame не осталось данных после очистки.")
            raise ValueError("В DataFrame не осталось данных после очистки. Проверьте формат данных в файле.")

        if fit_scaler or scaler is None:
            new_scaler = MinMaxScaler()
            scaled_data = new_scaler.fit_transform(data)
            self.update_status_signal.emit("📊 Данные масштабированы новым MinMaxScaler.")
            logger.info("Данные масштабированы новым MinMaxScaler.")
            return scaled_data, new_scaler
        else:
            scaled_data = scaler.transform(data)
            self.update_status_signal.emit("📊 Данные масштабированы существующим MinMaxScaler.")
            logger.info("Данные масштабированы существующим MinMaxScaler.")
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
            logger.warning("Попытка повторного запуска обучения.")
            return

        self.is_learning_running = True
        self.update_status_signal.emit("▶️ Начинаем обучение модели...")
        logger.info(
            f"Начало обучения. Файл: {file_path}, временной шаг: {time_step}, эпохи: {epochs}, батч: {batch_size}")

        try:
            scaled_data, self.scaler = self.load_and_preprocess_data(file_path, self.scaler, fit_scaler=True)
            X_train = self.create_dataset(scaled_data, time_step)

            logger.info(f"Создано {X_train.shape[0]} временных окон для обучения.")

            if X_train.size == 0:
                raise ValueError("Недостаточно данных для создания временных окон. Уменьшите 'Временной шаг'.")

            X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])
            self.autoencoder = self.build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))

            plot_callback = PlotCallback(self.update_plot_signal)

            history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                           verbose=0, callbacks=[plot_callback])
            logger.info("Обучение модели завершено.")

            self.update_status_signal.emit("🧠 Модель обучена. Расчет порога аномалий...")

            reconstruction_errors = np.mean(np.power(X_train - self.autoencoder.predict(X_train, verbose=0), 2),
                                            axis=(1, 2))
            threshold = np.percentile(reconstruction_errors, 95)
            logger.info(f"Рассчитан порог аномалий: {threshold:.4f}")

            results = {
                'threshold': threshold
            }

            self.learning_finished.emit(results)
            self.update_status_signal.emit("✅ Обучение завершено!")
        except Exception as e:
            self.update_status_signal.emit(f"❌ Произошла ошибка во время обучения: {e}")
            logger.error("Ошибка в рабочем потоке во время обучения", exc_info=True)
        finally:
            self.is_learning_running = False

    @QtCore.pyqtSlot(str, int, float)
    def start_testing(self, file_path, time_step, threshold):
        if self.is_testing_running:
            self.update_status_signal.emit("⚠️ Тестирование уже запущено.")
            logger.warning("Попытка повторного запуска тестирования.")
            return
        if self.autoencoder is None or self.scaler is None:
            self.update_status_signal.emit("⚠️ Ошибка: Модель не обучена или не загружена.")
            logger.error("Тестирование невозможно: модель не обучена или не загружена.")
            return

        self.is_testing_running = True
        self.update_status_signal.emit("▶️ Начинаем тестирование модели...")
        logger.info(f"Начало тестирования. Файл: {file_path}, временной шаг: {time_step}, порог: {threshold}")

        try:
            scaled_data, _ = self.load_and_preprocess_data(file_path, self.scaler, fit_scaler=False)
            logger.info(f"Всего строк в тестовом файле: {len(scaled_data)}")

            X_new = self.create_dataset(scaled_data, time_step)
            logger.info(f"Создано {X_new.shape[0]} временных окон для тестирования.")

            if X_new.size == 0:
                raise ValueError("Недостаточно данных для создания временных окон. Уменьшите 'Временной шаг'.")

            X_new = X_new.reshape(X_new.shape[0], time_step, scaled_data.shape[1])

            self.update_status_signal.emit("🔍 Выполняется предсказание на тестовых данных...")
            reconstruction_errors = np.mean(np.power(X_new - self.autoencoder.predict(X_new, verbose=0), 2),
                                            axis=(1, 2))
            anomalies = np.where(reconstruction_errors > threshold)[0]
            logger.info(f"Тестирование завершено. Обнаружено {len(anomalies)} аномалий.")

            results = {
                'reconstruction_errors': reconstruction_errors,
                'anomalies': anomalies
            }

            self.testing_finished.emit(results)
            self.update_status_signal.emit(f"✅ Тестирование завершено. Обнаружено {len(anomalies)} аномалий.")
        except Exception as e:
            self.update_status_signal.emit(f"❌ Произошла ошибка во время тестирования: {e}")
            logger.error("Ошибка в рабочем потоке во время тестирования", exc_info=True)
        finally:
            self.is_testing_running = False

    def _log_incident(self, error, metrics):
        """
        Создает и записывает инцидент в локальный файл в формате JSON.
        """
        # Создаем директорию 'incidents', если она не существует
        if not os.path.exists('incidents'):
            os.makedirs('incidents')

        # Формируем структуру JSON-сообщения согласно требованиям
        incident = {
            "gid": None,
            "sid": None,
            "rev": None,
            "signature_msg": f"Обнаружена аномалия. Ошибка реконструкции: {error:.4f}",
            "appearance_time": datetime.now().isoformat(),
            "priority": 1,
            # Временно заполняем неизвестными значениями, т.к. система не имеет доступа к пакетам
            "source_ip": "N/A",
            "source_port": "N/A",
            "destination_ip": "N/A",
            "destination_port": "N/A",
            "packet_dump": "N/A",
            "metrics": metrics
        }

        file_path = 'incidents/incidents.json'

        # Если файл не существует, создаем его с начальной структурой
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([incident], f, indent=4)
        else:
            # Читаем существующие инциденты, добавляем новый и перезаписываем файл
            try:
                with open(file_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    data.append(incident)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            except json.JSONDecodeError:
                # Если файл поврежден, создаем новый
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([incident], f, indent=4)

    @QtCore.pyqtSlot(list, int, float)
    def process_online_data(self, data_list, time_step, threshold):
        if self.autoencoder is None or self.scaler is None:
            self.update_status_signal.emit("❌ Ошибка: Модель не обучена или не загружена для онлайн-тестирования.")
            return

        try:
            if not data_list:
                raise ValueError("Получен пустой список для обработки.")

            # Преобразуем список в NumPy-массив
            data_array = np.array(data_list)

            # Убедимся, что количество признаков совпадает с обученной моделью
            if data_array.shape[1] != self.scaler.n_features_in_:
                raise ValueError(
                    f"Неверное количество признаков в онлайн-данных. Ожидается {self.scaler.n_features_in_}, получено {data_array.shape[1]}."
                )

            # Применяем тот же scaler, что использовался при обучении
            scaled_data = self.scaler.transform(data_array)

            # Создаем временное окно для предсказания
            X_new = scaled_data.reshape(1, time_step, scaled_data.shape[1])

            # Предсказание
            reconstruction_errors = np.mean(np.power(X_new - self.autoencoder.predict(X_new, verbose=0), 2),
                                            axis=(1, 2))
            is_anomaly = reconstruction_errors[0] > threshold

            results = {
                'error': reconstruction_errors[0],
                'is_anomaly': is_anomaly
            }
            self.online_results_signal.emit(results)

            # --- НОВАЯ ЛОГИКА ДЛЯ КОМБИНИРОВАННОГО ПОДХОДА ---
            # Если обнаружена аномалия, записываем ее в локальный файл
            if is_anomaly:
                self._log_incident(results['error'], data_list)
                self.update_status_signal.emit("📝 Инцидент записан в локальный журнал.")
            # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

        except Exception as e:
            logger.error(f"Ошибка при обработке онлайн-данных: {e}", exc_info=True)
            self.update_status_signal.emit(f"❌ Ошибка при обработке онлайн-данных: {e}")


# --- Новый класс для онлайн-тестирования (сетевой сокет) ---
class OnlineTestingWorker(QtCore.QObject):
    """
    Класс-работник для прослушивания порта и приема данных по сети.
    """
    update_status_signal = QtCore.pyqtSignal(str)
    # Сигнал для отправки полученных данных в MLWorker
    data_received_signal = QtCore.pyqtSignal(list, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self.socket = None
        self.data_buffer = collections.deque(maxlen=1000)
        self.lock = QtCore.QMutex()

    @QtCore.pyqtSlot(int, int, float)
    def start_listening(self, port, time_step, threshold):
        if self.is_running:
            self.update_status_signal.emit("⚠️ Онлайн-тестирование уже запущено.")
            return

        self.is_running = True
        self.time_step = time_step
        self.threshold = threshold

        self.update_status_signal.emit(f"▶️ Запуск сервера для приема данных на порту: {port}...")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(1.0)

        try:
            self.socket.bind(('127.0.0.1', port))
            self.socket.listen(1)
            self.update_status_signal.emit(f"✅ Сервер запущен, ожидаем подключения от сниффера...")

            conn, addr = self.socket.accept()
            with conn:
                self.update_status_signal.emit(f"🔗 Установлено соединение с клиентом: {addr}")
                buffer = b''
                while self.is_running:
                    data = conn.recv(4096)
                    if not data:
                        self.update_status_signal.emit(f"Соединение закрыто клиентом: {addr}")
                        break

                    buffer += data
                    # Попытка обработать данные.
                    try:
                        decoded_data = buffer.decode('utf-8')
                        data_list = json.loads(decoded_data)

                        if not isinstance(data_list, list):
                            self.update_status_signal.emit("❌ Получены некорректные данные: ожидался список.")
                            buffer = b''
                            continue

                        self.lock.lock()
                        try:
                            # Используем append(), чтобы добавить список целиком
                            self.data_buffer.append(data_list)

                            if len(self.data_buffer) >= self.time_step:
                                window = list(self.data_buffer)[-self.time_step:]
                                self.data_received_signal.emit(window, self.time_step, self.threshold)
                        finally:
                            self.lock.unlock()

                        buffer = b''

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Ошибка при обработке полученных данных: {e}", exc_info=True)
                        self.update_status_signal.emit(f"❌ Ошибка при обработке данных: {e}")
                        buffer = b''

        except socket.timeout:
            if self.is_running:
                QtCore.QTimer.singleShot(100, lambda: self.start_listening(port, time_step,
                                                                           threshold))
        except Exception as e:
            self.update_status_signal.emit(f"❌ Ошибка при запуске сервера: {e}")
            logger.critical("Не удалось запустить сокет-сервер", exc_info=True)
        finally:
            self.stop_listening()

    @QtCore.pyqtSlot()
    def stop_listening(self):
        if self.is_running:
            self.is_running = False
            self.update_status_signal.emit("⏹️ Остановка сервера...")
            if self.socket:
                self.socket.close()
                self.socket = None
                self.update_status_signal.emit("✅ Сервер остановлен.")