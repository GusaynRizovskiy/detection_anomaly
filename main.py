# -*- coding: utf-8 -*-
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import pyqtgraph as pg
import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDoubleSpinBox
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError as mse_loss

from form_of_network import Ui_Dialog
from worker import MLWorker, OnlineTestingWorker


# --- Класс основного приложения ---
class AutoencoderApp(QtWidgets.QDialog, Ui_Dialog):
    # Новые сигналы для запуска операций в рабочих потоках
    start_learning_signal = QtCore.pyqtSignal(str, int, int, int)
    start_testing_signal = QtCore.pyqtSignal(str, int, float)
    # Сигналы для управления сетевым прослушиванием
    start_online_listening_signal = QtCore.pyqtSignal(int, int, float)
    stop_online_listening_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Переопределяем виджет для порога, так как он должен быть float
        self.spinBox_porog_anomaly_value = QDoubleSpinBox()
        self.spinBox_porog_anomaly_value.setDecimals(4)
        self.spinBox_porog_anomaly_value.setSingleStep(0.001)
        self.spinBox_porog_anomaly_value.setRange(0, 1)
        layout_to_insert = self.findChild(QtWidgets.QVBoxLayout, "verticalLayout")
        if layout_to_insert:
            layout_to_insert.insertWidget(3, self.spinBox_porog_anomaly_value)
        old_spinbox = self.findChild(QtWidgets.QSpinBox, "spinBox_porog_anomaly_value")
        if old_spinbox:
            old_spinbox.deleteLater()

        # === Инициализация фонового изображения и флага ===
        self.background_image_path = "fon/picture_fon2.jpg"
        self.background_updated_on_startup = False
        self.update_background()

        # Инициализация переменных для хранения путей к файлам
        self.learning_file_path = None
        self.test_file_path = None

        # === Установка значений по умолчанию для spinBox ===
        self.spinBox_epochs_value.setValue(100)
        self.spinBox_batch_size_value.setValue(32)
        self.spinBox_timestep_value.setValue(10)
        self.spinBox_porog_anomaly_value.setValue(0.001)
        self.spinBox_online_port.setValue(12345)

        # === Настройка логирования ===
        if not os.path.exists('log'):
            os.makedirs('log')
        log_file_name = f"log/app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_name, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # === Настройка рабочих потоков ===
        self.ml_worker_thread = QtCore.QThread()
        self.ml_worker = MLWorker()
        self.ml_worker.moveToThread(self.ml_worker_thread)

        self.online_worker_thread = QtCore.QThread()
        self.online_worker = OnlineTestingWorker()
        self.online_worker.moveToThread(self.online_worker_thread)

        # Подключение сигналов и слотов
        self.ml_worker_thread.started.connect(lambda: logging.info("ML-поток запущен."))
        self.online_worker_thread.started.connect(lambda: logging.info("Онлайн-поток запущен."))

        # Связываем сигналы UI с ML-потоком
        self.start_learning_signal.connect(self.ml_worker.start_learning)
        self.start_testing_signal.connect(self.ml_worker.start_testing)
        self.ml_worker.learning_finished.connect(self.handle_learning_results)
        self.ml_worker.testing_finished.connect(self.handle_testing_results)

        # Связываем сигналы UI с онлайн-потоком
        self.start_online_listening_signal.connect(self.online_worker.start_listening)
        self.stop_online_listening_signal.connect(self.online_worker.stop_listening)

        # Связываем сигналы между рабочими потоками
        self.online_worker.data_received_signal.connect(self.ml_worker.process_online_data)
        self.ml_worker.online_results_signal.connect(self.handle_online_results)

        # Связываем сигналы статуса и графиков со всеми потоками
        self.ml_worker.update_status_signal.connect(self.update_status)
        self.online_worker.update_status_signal.connect(self.update_status)
        self.ml_worker.update_plot_signal.connect(self.update_learning_plot)

        # Запускаем потоки
        self.ml_worker_thread.start()
        self.online_worker_thread.start()

        # Инициализация графиков
        self.setup_plots()
        self.epoch_x = []
        self.train_loss_y = []
        self.val_loss_y = []
        self.online_reconstruction_errors = []

        # Привязка кнопок к функциям
        self.pushButton_load_file_for_learning.clicked.connect(self.load_learning_file)
        self.pushButton_learn_model.clicked.connect(self.start_learning)
        self.pushButton_save_model.clicked.connect(self.save_model)
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.pushButton_load_test_file.clicked.connect(self.load_test_file)
        self.pushButton_test_model.clicked.connect(self.start_testing)
        self.pushButton_vvod_data.clicked.connect(self.check_and_accept_parameters)

        # Новая привязка для онлайн-тестирования
        self.pushButton_start_online_testing.clicked.connect(self.start_online_testing)
        self.pushButton_stop_online_testing.clicked.connect(self.stop_online_testing)

        self.update_status("Программа запущена. Загрузите файлы для обучения или тестирования.")
        logging.info("Приложение запущено и готово к работе.")
        self.set_ui_state_initial()

    def set_ui_state_initial(self):
        """Устанавливает начальное состояние кнопок."""
        self.pushButton_load_file_for_learning.setEnabled(True)
        self.pushButton_learn_model.setEnabled(True)
        self.pushButton_save_model.setEnabled(False)
        self.pushButton_load_model.setEnabled(True)
        self.pushButton_load_test_file.setEnabled(False)
        self.pushButton_test_model.setEnabled(False)
        self.pushButton_vvod_data.setEnabled(True)
        self.pushButton_start_online_testing.setEnabled(True)
        self.pushButton_stop_online_testing.setEnabled(False)

    def set_ui_state_after_learning_or_loading(self):
        """Устанавливает состояние кнопок после обучения или загрузки модели."""
        self.pushButton_load_file_for_learning.setEnabled(False)
        self.pushButton_learn_model.setEnabled(False)
        self.pushButton_save_model.setEnabled(True)
        self.pushButton_load_model.setEnabled(False)
        self.pushButton_load_test_file.setEnabled(True)
        self.pushButton_test_model.setEnabled(True)
        self.pushButton_vvod_data.setEnabled(False)
        self.pushButton_start_online_testing.setEnabled(True)
        self.pushButton_stop_online_testing.setEnabled(False)

    def closeEvent(self, event):
        self.ml_worker_thread.quit()
        self.ml_worker_thread.wait()
        self.online_worker_thread.quit()
        self.online_worker_thread.wait()
        event.accept()

    def resizeEvent(self, event):
        self.update_background()
        super().resizeEvent(event)

    def update_background(self):
        try:
            if os.path.exists(self.background_image_path):
                palette = QPalette()
                pixmap = QPixmap(self.background_image_path)
                scaled_pixmap = pixmap.scaled(self.size(),
                                              QtCore.Qt.KeepAspectRatioByExpanding,
                                              QtCore.Qt.SmoothTransformation)
                brush = QBrush(scaled_pixmap)
                palette.setBrush(QPalette.Background, brush)
                self.setPalette(palette)
                self.setAutoFillBackground(True)

                if not self.background_updated_on_startup:
                    logging.info("Фоновое изображение успешно обновлено и масштабировано.")
                    self.background_updated_on_startup = True
            else:
                if not self.background_updated_on_startup:
                    logging.warning(
                        f"Фоновое изображение не найдено по пути: {self.background_image_path}. Фон не будет установлен.")
                    self.background_updated_on_startup = True
        except Exception as e:
            if not self.background_updated_on_startup:
                logging.error(f"Ошибка при загрузке фонового изображения '{self.background_image_path}': {e}",
                              exc_info=True)
                QMessageBox.critical(self, "Ошибка загрузки фона", f"Не удалось загрузить фоновое изображение: {e}")
                self.background_updated_on_startup = True

    # --- Функции для GUI ---
    def update_status(self, message):
        self.text_zone.appendPlainText(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def setup_plots(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_loss = pg.PlotWidget()
        layout_loss = QtWidgets.QVBoxLayout(self.widget_plot_loss)
        layout_loss.addWidget(self.plot_loss)
        self.plot_loss.setTitle("Динамика ошибки: Обучение vs. Валидация")
        self.plot_loss.setLabel('left', 'Loss (MSE)')
        self.plot_loss.setLabel('bottom', 'Эпохи')
        self.curve_train_loss = self.plot_loss.plot(pen='b', name='Обучение')
        self.curve_val_loss = self.plot_loss.plot(pen='r', name='Валидация')

        self.plot_reconstruction_error = pg.PlotWidget()
        layout_reconstruction = QtWidgets.QVBoxLayout(self.widget_plot_reconstruction_error)
        layout_reconstruction.addWidget(self.plot_reconstruction_error)
        self.plot_reconstruction_error.setTitle("Ошибка реконструкции (MSE) по временным окнам")
        self.plot_reconstruction_error.setLabel('left', 'MSE')
        self.plot_reconstruction_error.setLabel('bottom', 'Временные окна')
        self.curve_reconstruction_error = self.plot_reconstruction_error.plot(pen='g', name='MSE')
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.plot_reconstruction_error.addItem(self.threshold_line)

        self.plot_anomaly_comparison = pg.PlotWidget()
        layout_comparison = QtWidgets.QVBoxLayout(self.widget_plot_anomaly_comparison)
        layout_comparison.addWidget(self.plot_anomaly_comparison)
        self.plot_anomaly_comparison.setTitle("Сравнение: Предсказания vs. Истинные аномалии")
        self.plot_anomaly_comparison.setLabel('left', 'Значение')
        self.plot_anomaly_comparison.setLabel('bottom', 'Временные окна')
        self.curve_true_anomalies = self.plot_anomaly_comparison.plot(pen='b', name='Истинные аномалии')
        self.curve_predicted_anomalies = self.plot_anomaly_comparison.plot(pen='r', name='Предсказания модели')

    @QtCore.pyqtSlot(dict)
    def update_learning_plot(self, epoch_logs):
        if not self.ml_worker.is_learning_running:
            return

        self.epoch_x.append(epoch_logs['epoch'])
        self.train_loss_y.append(epoch_logs['loss'])
        self.val_loss_y.append(epoch_logs['val_loss'])
        self.curve_train_loss.setData(self.epoch_x, self.train_loss_y)
        self.curve_val_loss.setData(self.epoch_x, self.val_loss_y)
        self.update_status(
            f"Эпоха {epoch_logs['epoch']}: Loss = {epoch_logs['loss']:.4f}, Val Loss = {epoch_logs['val_loss']:.4f}")

    def load_learning_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл для обучения", "", "CSV Files (*.csv)")
        if file_path:
            self.learning_file_path = file_path
            self.update_status(f"Файл для обучения выбран: {file_path}")

    def load_test_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл для тестирования", "", "CSV Files (*.csv)")
        if file_path:
            self.test_file_path = file_path
            self.update_status(f"Файл для тестирования выбран: {file_path}")

    def check_and_accept_parameters(self):
        time_step = self.spinBox_timestep_value.value()
        epochs = self.spinBox_epochs_value.value()
        batch_size = self.spinBox_batch_size_value.value()
        threshold = self.spinBox_porog_anomaly_value.value()

        if time_step == 0 or epochs == 0 or batch_size == 0:
            QMessageBox.warning(self, "Ошибка ввода",
                                "Значения 'Временной шаг', 'Количество эпох' и 'Размер батча' не могут быть равны 0.")
        else:
            self.update_status("✅ Параметры успешно введены и приняты.")
            QMessageBox.information(self, "Успешный ввод", "Параметры обучения и тестирования приняты.")

    def start_learning(self):
        if not self.learning_file_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала выберите файл для обучения.")
            return

        time_step = self.spinBox_timestep_value.value()
        epochs = self.spinBox_epochs_value.value()
        batch_size = self.spinBox_batch_size_value.value()

        if time_step == 0 or epochs == 0 or batch_size == 0:
            QMessageBox.warning(self, "Ошибка",
                                "Значения 'Временной шаг', 'Количество эпох' и 'Размер батча' не могут быть равны 0. Пожалуйста, введите корректные значения.")
            return

        # Блокировка кнопок во время обучения
        self.pushButton_load_file_for_learning.setEnabled(False)
        self.pushButton_learn_model.setEnabled(False)
        self.pushButton_load_model.setEnabled(False)
        self.pushButton_save_model.setEnabled(False)

        self.epoch_x.clear()
        self.train_loss_y.clear()
        self.val_loss_y.clear()
        self.curve_train_loss.setData([], [])
        self.curve_val_loss.setData([], [])

        self.start_learning_signal.emit(self.learning_file_path, time_step, epochs, batch_size)

    def handle_learning_results(self, results):
        self.threshold = results['threshold']
        self.spinBox_porog_anomaly_value.setValue(self.threshold)
        self.update_status(f"Автоматически рассчитанный порог аномалии: {self.threshold:.4f}")

        # Разблокировка кнопок после обучения
        self.set_ui_state_after_learning_or_loading()

    def save_model(self):
        if self.ml_worker.autoencoder is None or self.ml_worker.scaler is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите или загрузите модель.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.ml_worker.autoencoder.save(file_path)
                with open(file_path.replace('.h5', '_scaler.pkl'), 'wb') as f:
                    pickle.dump(self.ml_worker.scaler, f)
                self.update_status(f"✅ Модель и скейлер успешно сохранены в: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить модель: {e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                if not os.path.exists(file_path):
                    QMessageBox.critical(self, "Ошибка загрузки", "Файл модели не найден по указанному пути.")
                    return

                custom_objects = {
                    'mse': mse_loss,
                    'MeanSquaredError': MeanSquaredError
                }
                self.ml_worker.autoencoder = load_model(file_path, custom_objects=custom_objects)

                scaler_path = file_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.ml_worker.scaler = pickle.load(f)
                    self.update_status(f"✅ Модель успешно загружена из: {file_path}")
                    self.update_status("✅ Скейлер для нормализации успешно загружен.")
                    self.set_ui_state_after_learning_or_loading()
                else:
                    QMessageBox.warning(self, "Предупреждение", "Файл скейлера не найден. Загружена только модель.")
                    self.ml_worker.scaler = None
                    self.update_status(f"✅ Модель успешно загружена из: {file_path}, но скейлер не найден.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить модель: {e}")
                logging.error("Ошибка при загрузке модели", exc_info=True)
                self.set_ui_state_initial()

    def start_testing(self):
        if self.ml_worker.autoencoder is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите или загрузите модель.")
            return
        if not self.test_file_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала выберите файл для тестирования.")
            return

        time_step = self.spinBox_timestep_value.value()
        threshold = self.spinBox_porog_anomaly_value.value()

        if time_step == 0:
            QMessageBox.warning(self, "Ошибка",
                                "Значение 'Временной шаг' не может быть равно 0. Пожалуйста, введите корректное значение.")
            return

        if threshold == 0:
            QMessageBox.warning(self, "Предупреждение",
                                "Порог аномалии равен 0. Используйте рассчитанный порог или введите вручную.")

        self.start_testing_signal.emit(self.test_file_path, time_step, threshold)

    def handle_testing_results(self, results):
        threshold = self.spinBox_porog_anomaly_value.value()
        self.curve_reconstruction_error.setData(results['reconstruction_errors'])
        self.threshold_line.setPos(threshold)
        anomaly_flags = np.zeros(len(results['reconstruction_errors']))
        anomaly_flags[results['anomalies']] = 1
        self.curve_predicted_anomalies.setData(anomaly_flags)
        if len(results['anomalies']) > 0:
            self.update_status("--- Индексы аномальных временных окон: ---")
            self.update_status(f"{results['anomalies']}")

    # --- Новые методы для онлайн-тестирования ---
    def start_online_testing(self):
        if self.ml_worker.autoencoder is None or self.ml_worker.scaler is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите или загрузите модель.")
            return
        if self.online_worker.is_running:
            QMessageBox.warning(self, "Предупреждение", "Онлайн-тестирование уже запущено.")
            return

        port = self.spinBox_online_port.value()
        time_step = self.spinBox_timestep_value.value()
        threshold = self.spinBox_porog_anomaly_value.value()

        self.online_reconstruction_errors.clear()
        self.plot_reconstruction_error.clear()
        self.curve_reconstruction_error = self.plot_reconstruction_error.plot(pen='g', name='MSE')
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.plot_reconstruction_error.addItem(self.threshold_line)
        self.threshold_line.setPos(threshold)

        # Блокировка кнопок, пока идет онлайн-тестирование
        self.pushButton_start_online_testing.setEnabled(False)
        self.pushButton_stop_online_testing.setEnabled(True)

        self.start_online_listening_signal.emit(port, time_step, threshold)

    def stop_online_testing(self):
        self.stop_online_listening_signal.emit()
        self.pushButton_start_online_testing.setEnabled(True)
        self.pushButton_stop_online_testing.setEnabled(False)

    @QtCore.pyqtSlot(dict)
    def handle_online_results(self, results):
        error = results['error']
        is_anomaly = results['is_anomaly']

        self.online_reconstruction_errors.append(error)
        self.curve_reconstruction_error.setData(self.online_reconstruction_errors)

        if is_anomaly:
            self.update_status(f"⚠️ АНОМАЛИЯ ОБНАРУЖЕНА! Ошибка: {error:.4f}")
            x_pos = len(self.online_reconstruction_errors) - 1
            anomaly_point = pg.ScatterPlotItem([x_pos], [error], symbol='o', size=10, pen={'color': 'r', 'width': 2},
                                               brush='r')
            self.plot_reconstruction_error.addItem(anomaly_point)
        else:
            self.update_status(f"✅ Данные получены. Ошибка: {error:.4f}")


# --- Главная точка входа ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AutoencoderApp()
    window.showMaximized()
    sys.exit(app.exec_())