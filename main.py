# -*- coding: utf-8 -*-

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from tensorflow.keras.models import load_model
import pickle

# Импортируем класс UI-формы из модуля form_of_network и новый класс Worker
from form_of_network import Ui_Dialog
from worker import Worker  # Импортируем новый класс

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

        # === Настройка рабочего потока ===
        self.worker_thread = QtCore.QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)

        # Подключение сигналов и слотов
        self.worker_thread.started.connect(lambda: logging.info("Рабочий поток запущен."))
        self.worker_thread.finished.connect(lambda: logging.info("Рабочий поток завершен."))

        self.worker.learning_finished.connect(self.handle_learning_results)
        self.worker.testing_finished.connect(self.handle_testing_results)
        self.worker.update_status_signal.connect(self.update_status)

        self.worker_thread.start()

        # Инициализация графиков
        self.setup_plots()

        # Привязка кнопок к функциям
        self.pushButton_load_file_for_learning.clicked.connect(self.load_learning_file)
        self.pushButton_learn_model.clicked.connect(self.start_learning)
        self.pushButton_save_model.clicked.connect(self.save_model)
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.pushButton_load_test_file.clicked.connect(self.load_test_file)
        self.pushButton_test_model.clicked.connect(self.start_testing)
        # !!! НОВАЯ СТРОКА: Подключаем новую кнопку !!!
        self.pushButton_vvod_data.clicked.connect(self.check_and_accept_parameters)
        # self.pushButton_parametr_input.clicked.connect(self.accept_parameters) # Удалена несуществующая кнопка

        # Вывод приветственного сообщения
        self.update_status("Программа запущена. Загрузите файлы для обучения или тестирования.")

    # Переопределяем метод закрытия, чтобы безопасно остановить поток
    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

    def resizeEvent(self, event):
        """Обработчик события изменения размера окна."""
        self.update_background()
        super().resizeEvent(event)

    def update_background(self):
        """Метод для установки и масштабирования фонового изображения, с проверкой для консоли."""
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
        """Обновляет текстовое поле статуса."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_zone.appendPlainText(f"[{timestamp}] {message}")

    def setup_plots(self):
        """Настраивает виджеты для графиков."""

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

    def check_and_accept_parameters(self):
        """Проверяет заполненность полей и принимает параметры."""
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

    # --- Функции-слоты для управления рабочим потоком ---
    def start_learning(self):
        """Запускает процесс обучения с проверкой параметров."""
        if not self.learning_file_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала выберите файл для обучения.")
            return

        time_step = self.spinBox_timestep_value.value()
        epochs = self.spinBox_epochs_value.value()
        batch_size = self.spinBox_batch_size_value.value()

        # Проверяем, что параметры не равны нулю перед началом обучения
        if time_step == 0 or epochs == 0 or batch_size == 0:
            QMessageBox.warning(self, "Ошибка",
                                "Значения 'Временной шаг', 'Количество эпох' и 'Размер батча' не могут быть равны 0. Пожалуйста, введите корректные значения.")
            return

        # Отправляем параметры в рабочий поток
        self.worker.start_learning.emit(self.learning_file_path, time_step, epochs, batch_size)

    def handle_learning_results(self, results):
        """Слот для обработки результатов обучения из рабочего потока."""
        self.threshold = results['threshold']
        self.spinBox_porog_anomaly_value.setValue(self.threshold)
        self.update_status(f"Автоматически рассчитанный порог аномалии: {self.threshold:.4f}")

        self.curve_train_loss.setData(results['loss'])
        self.curve_val_loss.setData(results['val_loss'])

    def save_model(self):
        """Сохраняет обученную модель и scaler."""
        if self.worker.autoencoder is None or self.worker.scaler is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите или загрузите модель.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.worker.autoencoder.save(file_path)
                with open(file_path.replace('.h5', '_scaler.pkl'), 'wb') as f:
                    pickle.dump(self.worker.scaler, f)
                self.update_status(f"✅ Модель и скейлер успешно сохранены в: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить модель: {e}")

    def load_model(self):
        """Загружает ранее обученную модель."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.worker.autoencoder = load_model(file_path)
                self.update_status(f"✅ Модель успешно загружена из: {file_path}")
                with open(file_path.replace('.h5', '_scaler.pkl'), 'rb') as f:
                    self.worker.scaler = pickle.load(f)
                self.update_status("✅ Скейлер для нормализации успешно загружен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить модель: {e}")

    def start_testing(self):
        """Запускает процесс тестирования с проверкой параметров."""
        if self.worker.autoencoder is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите или загрузите модель.")
            return
        if not self.test_file_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала выберите файл для тестирования.")
            return

        time_step = self.spinBox_timestep_value.value()
        threshold = self.spinBox_porog_anomaly_value.value()

        # Проверяем, что параметры не равны нулю перед началом тестирования
        if time_step == 0:
            QMessageBox.warning(self, "Ошибка",
                                "Значение 'Временной шаг' не может быть равно 0. Пожалуйста, введите корректное значение.")
            return

        if threshold == 0:
            QMessageBox.warning(self, "Предупреждение",
                                "Порог аномалии равен 0. Используйте рассчитанный порог или введите вручную.")

        self.worker.start_testing.emit(self.test_file_path, time_step, threshold)

    def handle_testing_results(self, results):
        """Слот для обработки результатов тестирования из рабочего потока."""
        threshold = self.spinBox_porog_anomaly_value.value()

        self.curve_reconstruction_error.setData(results['reconstruction_errors'])
        self.threshold_line.setPos(threshold)

        anomaly_flags = np.zeros(len(results['reconstruction_errors']))
        anomaly_flags[results['anomalies']] = 1
        self.curve_predicted_anomalies.setData(anomaly_flags)

        if len(results['anomalies']) > 0:
            self.update_status("--- Индексы аномальных временных окон: ---")
            self.update_status(f"{results['anomalies']}")


# --- Основная точка входа в приложение ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AutoencoderApp()
    window.showMaximized()
    sys.exit(app.exec_())