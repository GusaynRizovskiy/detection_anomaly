from datetime import datetime

from django.contrib.sites import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten, RepeatVector
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# === 1. Загрузка и предварительная обработка данных ===
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter=';', encoding='cp1251')
    data.columns = data.columns.str.strip()

    if 'Время захвата пакетов' in data.columns:
        data.drop(columns=['Время захвата пакетов'], inplace=True)

    data = data.replace(',', '.', regex=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Нормализация данных
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data


# === 2. Создание временных рядов ===
def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), :]
        X.append(a)
    return np.array(X)


# === 3. CNN-LSTM Autoencoder Model ===
def build_cnn_lstm_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    conv_encoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
    lstm_encoder = LSTM(50, activation='relu', return_sequences=False)(conv_encoder)

    # Decoder
    repeat = RepeatVector(input_shape[0])(lstm_encoder)
    lstm_decoder = LSTM(50, activation='relu', return_sequences=True)(repeat)
    conv_decoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(lstm_decoder)
    output_layer = Conv1D(input_shape[1], kernel_size=3, padding='same')(conv_decoder)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model


# === 4. Обучение модели на исходных данных ===
try:
    # Загрузка и подготовка исходных данных
    scaled_data = load_and_preprocess_data('learning_data.csv')

    # Создание временных рядов
    time_step = 10
    X_train = create_dataset(scaled_data, time_step)

    # Reshape X to (samples, time_steps, features)
    X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])

    print("Shape of input dataset:", X_train.shape)

    # Building model
    autoencoder = build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))

    # Model Training
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2)

    print("Model trained successfully.")
except Exception as e:
    print(f"Error building or training the model: {e}")
    exit()

# === 5. Обнаружение аномалий в новом файле ===
# Работа с файлом, который содержит аномалии
try:
    # Загрузка и подготовка новых данных с аномалиями
    new_scaled_data = load_and_preprocess_data('proverka_data.csv')

    # Создание временных рядов для новых данных
    X_new = create_dataset(new_scaled_data, time_step)

    # Reshape X to (samples, time_steps, features)
    X_new = X_new.reshape(X_new.shape[0], time_step, new_scaled_data.shape[1])

    # Загрузка заранее вычисленного порога (или задайте вручную)
    # Например, threshold = 0.01
    threshold = ...  # Загрузите или задайте порог

    for idx in range(X_new.shape[0]):
        sample = X_new[idx:idx+1]  # Выделяем одно окно с нужной размерностью
        reconstructed_sample = autoencoder.predict(sample)
        mse_value = np.mean(np.power(sample - reconstructed_sample, 2))

        if mse_value > threshold:
            anomaly_event = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "anomaly_type": "anomaly",
                "anomaly_score": float(mse_value),
                "description": "Обнаружена аномалия автоэнкодером. Ошибка реконструкции превышает порог.",
                "job_id": "autoencoder_ddos_2025",
                "sequence_index": int(idx),
                "mse_value": float(mse_value),
                "status": "THREAT",
                "host": "server01",
                "version": "1.0"
            }
            # Здесь вызовите функцию отправки в SIEM, например:
            send_to_siem(anomaly_event)


        # Отправка данных в SIEM систему
        def send_to_siem(json_data):
            """
            Отправляет событие аномалии в SIEM-систему через HTTP POST.

            Args:
                json_data (dict): Словарь с данными аномалии, который будет преобразован в JSON.
            """
            url = "https://your-siem-api.example.com/events"  # Замените на URL вашего SIEM API
            headers = {'Content-Type': 'application/json'}

            try:
                response = requests.post(url, headers=headers, data=json.dumps(json_data))
                if response.status_code == 200:
                    print(f"Аномалия успешно отправлена: sequence_index={json_data.get('sequence_index')}")
                else:
                    print(f"Ошибка отправки в SIEM: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Ошибка при отправке в SIEM: {e}")

except ValueError as e:
    print(f"Anomaly detection error: {e}")
except Exception as e:
    print(f"Error during anomaly detection: {e}")


