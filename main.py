from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten, RepeatVector,Conv1DTranspose
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
# Пример использования
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === 1. Загрузка данных ===
data = pd.read_csv('data.csv', delimiter=';', encoding='cp1251')

# === 2. Предварительная обработка данных ===
data.columns = data.columns.str.strip()

if 'Время захвата пакетов' in data.columns:
    data.drop(columns=['Время захвата пакетов'], inplace=True)

data = data.replace(',', '.', regex=True)
data = data.apply(pd.to_numeric, errors='coerce')

# Нормализация данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# === 3. Создание временных рядов ===
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, :])  # Ensure Y is also created
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_data, time_step)

# Reshape X to (samples, time_steps, features)
X = X.reshape(X.shape[0], time_step, data.shape[1])

print("Shape of input dataset:", X.shape)

# === 4. CNN-LSTM Autoencoder Model ===
def build_cnn_lstm_autoencoder(input_shape):
    """
    Автоэнкодер для 3D данных (timesteps, features).
    :param input_shape: Кортеж (timesteps, features)
    """
    input_layer = Input(shape=input_shape)

    # Encoder
    # Создает одномерный светрочный слой и применяет его к входным данным(входному слою)
    conv_encoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
    lstm_encoder = LSTM(50, activation='relu', return_sequences=False)(conv_encoder)
    #returen_sequences = False возвращается только последнее выходное состояние, а не вся последовательность выходов.

    # Decoder
    repeat = RepeatVector(input_shape[0])(lstm_encoder)  # Восстанавливаем временные шаги
    lstm_decoder = LSTM(50, activation='relu', return_sequences=True)(repeat)
    conv_decoder = Conv1D(32, kernel_size=3, activation='relu', padding='same')(lstm_decoder)
    output_layer = Conv1D(input_shape[1], kernel_size=3, padding='same')(conv_decoder)  # Выходная форма = входной

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

try:
    # Building model
    #X.shape[1] - timesteps, X.shape[2] - features. Передаем значения в виде кортежа input_shape
    autoencoder = build_cnn_lstm_autoencoder((X.shape[1], X.shape[2]))

    # Model Training
    autoencoder.fit(X, X, epochs=20, batch_size=32, validation_split=0.2)  # Train to reconstruct the entire sequence

    print("Model trained successfully.")
except Exception as e:
    print(f"Error building or training the model: {e}")
    exit()

# === 5. Anomaly Detection ===
try:
    reconstructed_data = autoencoder.predict(X)

    # Вычислить ошибку реконструкции (MSE) для каждой последовательности
    mse = np.mean(np.power(X - reconstructed_data, 2), axis=(1,2)) # Усреднение по временным шагам и признакам

    # Установить порог для обнаружения аномалий
    threshold = np.mean(mse) + 3 * np.std(mse)

    # Идентифицировать аномалии
    anomalies = mse > threshold

    # Вывести индексы аномалий
    print("Аномалии найдены по индексам:", np.where(anomalies)[0])
except ValueError as e:
    print(f"Anomaly detection error: {e}")
except Exception as e:
    print(f"Error during anomaly detection: {e}")



