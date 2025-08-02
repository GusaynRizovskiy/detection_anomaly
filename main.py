# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import pyqtgraph as pg
import requests
import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDoubleSpinBox
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector

# Эти строки нужно разместить как можно раньше,
# чтобы они вступили в силу до первого импорта TensorFlow.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Импортируем класс UI-формы из модуля form_of_network
from form_of_network import Ui_Dialog

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Функции для работы с данными и моделью ===
def load_and_preprocess_data(file_path, scaler, fit_scaler=True):
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


def create_dataset(data, time_step):
    """Создание временных рядов."""
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), :]
        X.append(a)
    return np.array(X)


def build_cnn_lstm_autoencoder(input_shape):
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


def send_to_siem(json_data):
    """Отправляет событие аномалии в SIEM-систему через HTTP POST."""
    url = "https://your-siem-api.example.com/events"  # Замените на URL
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(json_data))
        if response.status_code == 200:
            logging.info(f"Аномалия успешно отправлена: sequence_index={json_data.get('sequence_index')}")
        else:
            logging.error(f"Ошибка отправки в SIEM: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при отправке в SIEM: {e}", exc_info=True)


# --- Класс основного приложения ---
class AutoencoderApp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Переопределяем виджет для порога, так как он должен быть float
        self.spinBox_porog_anomaly_value = QDoubleSpinBox()
        self.spinBox_porog_anomaly_value.setDecimals(4)
        self.spinBox_porog_anomaly_value.setSingleStep(0.001)
        self.spinBox_porog_anomaly_value.setRange(0, 1)
        self.verticalLayout.insertWidget(3, self.spinBox_porog_anomaly_value)
        self.findChild(QtWidgets.QSpinBox, "spinBox_porog_anomaly_value").deleteLater()

        # === Загрузка фонового изображения с обработкой ошибок ===
        background_image_path = "fon/bg.jpg"  # Укажите здесь имя вашего файла
        try:
            if os.path.exists(background_image_path):
                palette = QPalette()
                pixmap = QPixmap(background_image_path)
                palette.setBrush(QPalette.Background, QBrush(pixmap.scaled(self.size(), QtCore.Qt.IgnoreAspectRatio)))
                self.setPalette(palette)
                self.setAutoFillBackground(True)
                logging.info(f"Фоновое изображение успешно загружено: {background_image_path}")
            else:
                logging.warning(
                    f"Фоновое изображение не найдено по пути: {background_image_path}. Фон не будет установлен.")
        except Exception as e:
            logging.error(f"Ошибка при загрузке фонового изображения '{background_image_path}': {e}", exc_info=True)
            QMessageBox.critical(self, "Ошибка загрузки фона", f"Не удалось загрузить фоновое изображение: {e}")

        # Инициализация переменных
        self.autoencoder = None
        self.scaler = None
        self.threshold = None
        self.learning_file_path = None
        self.test_file_path = None
        self.train_loss_history = []
        self.val_loss_history = []

        # Инициализация графиков
        self.setup_plots()

        # Привязка кнопок к функциям
        self.pushButton_load_file_for_learning.clicked.connect(self.load_learning_file)
        self.pushButton_learn_model.clicked.connect(self.learn_model)
        self.pushButton_save_model.clicked.connect(self.save_model)
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.pushButton_load_test_file.clicked.connect(self.load_test_file)
        self.pushButton_test_model.clicked.connect(self.test_model)

        # Вывод приветственного сообщения
        self.update_status("Программа запущена. Загрузите файлы для обучения или тестирования.")

    # --- Функции для GUI ---
    def update_status(self, message):
        """Обновляет текстовое поле статуса."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_zone.appendPlainText(f"[{timestamp}] {message}")

    def setup_plots(self):
        """Настраивает виджеты для графиков."""

        # === Установка белого фона для всех графиков ===
        # Используем pg.setConfigOption для установки глобальных параметров
        pg.setConfigOption('background', 'w')  # 'w' - белый цвет
        pg.setConfigOption('foreground', 'k')  # 'k' - черный цвет для текста и осей

        # График ошибки обучения и валидации
        self.plot_loss = pg.PlotWidget()
        layout_loss = QtWidgets.QVBoxLayout(self.widget_plot_loss)
        layout_loss.addWidget(self.plot_loss)
        self.plot_loss.setTitle("Динамика ошибки: Обучение vs. Валидация")
        self.plot_loss.setLabel('left', 'Loss (MSE)')
        self.plot_loss.setLabel('bottom', 'Эпохи')
        self.curve_train_loss = self.plot_loss.plot(pen='b', name='Обучение')
        self.curve_val_loss = self.plot_loss.plot(pen='r', name='Валидация')

        # График ошибки реконструкции
        self.plot_reconstruction_error = pg.PlotWidget()
        layout_reconstruction = QtWidgets.QVBoxLayout(self.widget_plot_reconstruction_error)
        layout_reconstruction.addWidget(self.plot_reconstruction_error)
        self.plot_reconstruction_error.setTitle("Ошибка реконструкции (MSE) по временным окнам")
        self.plot_reconstruction_error.setLabel('left', 'MSE')
        self.plot_reconstruction_error.setLabel('bottom', 'Временные окна')
        self.curve_reconstruction_error = self.plot_reconstruction_error.plot(pen='g', name='MSE')
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.plot_reconstruction_error.addItem(self.threshold_line)

        # График сравнения аномалий
        self.plot_anomaly_comparison = pg.PlotWidget()
        layout_comparison = QtWidgets.QVBoxLayout(self.widget_plot_anomaly_comparison)
        layout_comparison.addWidget(self.plot_anomaly_comparison)
        self.plot_anomaly_comparison.setTitle("Сравнение: Предсказания vs. Истинные аномалии")
        self.plot_anomaly_comparison.setLabel('left', 'Значение')
        self.plot_anomaly_comparison.setLabel('bottom', 'Временные окна')
        self.curve_true_anomalies = self.plot_anomaly_comparison.plot(pen='b', name='Истинные аномалии')
        self.curve_predicted_anomalies = self.plot_anomaly_comparison.plot(pen='r', name='Предсказания модели')

    # --- Функции для работы с файлами ---
    def load_learning_file(self):
        """Загружает файл для обучения."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл для обучения", "", "CSV Files (*.csv)")
        if file_path:
            self.learning_file_path = file_path
            self.update_status(f"Файл для обучения выбран: {file_path}")

    def load_test_file(self):
        """Загружает файл для тестирования."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл для тестирования", "", "CSV Files (*.csv)")
        if file_path:
            self.test_file_path = file_path
            self.update_status(f"Файл для тестирования выбран: {file_path}")

    # --- Функции для кнопок GUI ---
    def learn_model(self):
        """Обучает модель по нажатию кнопки."""
        try:
            if not self.learning_file_path:
                self.update_status("⚠️ Ошибка: Сначала загрузите файл для обучения.")
                return

            self.update_status("▶️ Начинаем обучение модели...")

            time_step = self.spinBox_timestep_value.value()
            epochs = self.spinBox_epochs_value.value()
            batch_size = self.spinBox_batch_size_value.value()

            scaled_data, self.scaler = load_and_preprocess_data(self.learning_file_path, self.scaler, fit_scaler=True)
            X_train = create_dataset(scaled_data, time_step)
            X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])

            self.update_status(f"Размер обучающего датасета: {X_train.shape}")
            self.autoencoder = build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))

            history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                           verbose=0)

            # Автоматический расчет порога
            reconstruction_errors = np.mean(np.power(X_train - self.autoencoder.predict(X_train), 2), axis=(1, 2))
            self.threshold = np.percentile(reconstruction_errors, 95)
            self.spinBox_porog_anomaly_value.setValue(self.threshold)

            self.update_status("✅ Модель успешно обучена!")
            self.update_status(f"Автоматически рассчитанный порог аномалии: {self.threshold:.4f}")

            # Визуализация ошибки обучения
            self.curve_train_loss.setData(history.history['loss'])
            self.curve_val_loss.setData(history.history['val_loss'])

        except Exception as e:
            self.update_status(f"❌ Произошла ошибка во время обучения: {e}")
            QMessageBox.critical(self, "Ошибка обучения", f"Произошла ошибка: {e}")

    def save_model(self):
        """Сохраняет обученную модель и scaler."""
        if self.autoencoder is None or self.scaler is None:
            self.update_status("⚠️ Ошибка: Сначала обучите или загрузите модель.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            self.autoencoder.save(file_path)
            # Сохраняем также scaler, так как он нужен для предобработки новых данных
            import pickle
            with open(file_path.replace('.h5', '_scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            self.update_status(f"✅ Модель и скейлер успешно сохранены в: {file_path}")

    def load_model(self):
        """Загружает ранее обученную модель."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.autoencoder = load_model(file_path)
                self.update_status(f"✅ Модель успешно загружена из: {file_path}")
                # Загружаем также scaler
                import pickle
                with open(file_path.replace('.h5', '_scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
                self.update_status("✅ Скейлер для нормализации успешно загружен.")
            except Exception as e:
                self.update_status(f"❌ Ошибка загрузки модели: {e}")

    def test_model(self):
        """Тестирует модель по нажатию кнопки."""
        try:
            if self.autoencoder is None:
                self.update_status("⚠️ Ошибка: Сначала обучите или загрузите модель.")
                return
            if not self.test_file_path:
                self.update_status("⚠️ Ошибка: Загрузите файл для тестирования.")
                return

            self.update_status("▶️ Начинаем тестирование модели...")

            time_step = self.spinBox_timestep_value.value()
            threshold = self.spinBox_porog_anomaly_value.value()
            if threshold == 0:
                self.update_status(
                    "⚠️ Предупреждение: Порог аномалии равен 0. Используйте рассчитанный порог или введите вручную.")

            scaled_data, _ = load_and_preprocess_data(self.test_file_path, self.scaler, fit_scaler=False)
            X_new = create_dataset(scaled_data, time_step)
            X_new = X_new.reshape(X_new.shape[0], time_step, scaled_data.shape[1])

            # Предсказание и расчет ошибок
            reconstruction_errors = np.mean(np.power(X_new - self.autoencoder.predict(X_new), 2), axis=(1, 2))
            anomalies = np.where(reconstruction_errors > threshold)[0]

            # Визуализация ошибок
            self.curve_reconstruction_error.setData(reconstruction_errors)
            self.threshold_line.setPos(threshold)

            # Визуализация аномалий (просто помечаем, где mse > threshold)
            anomaly_flags = np.zeros(len(reconstruction_errors))
            anomaly_flags[anomalies] = 1
            self.curve_predicted_anomalies.setData(anomaly_flags)

            self.update_status(f"✅ Тестирование завершено. Обнаружено {len(anomalies)} аномалий.")
            if len(anomalies) > 0:
                self.update_status("--- Индексы аномальных временных окон: ---")
                self.update_status(f"{anomalies}")
        except Exception as e:
            self.update_status(f"❌ Произошла ошибка во время тестирования: {e}")
            QMessageBox.critical(self, "Ошибка тестирования", f"Произошла ошибка: {e}")


# --- Основная точка входа в приложение ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AutoencoderApp()
    window.show()
    sys.exit(app.exec_())