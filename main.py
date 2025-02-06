from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten, RepeatVector
from keras.layers import Conv1DTranspose
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    input_layer = Input(shape=input_shape)

    # Encoder
    conv_layer = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(input_layer)
    flatten_layer = TimeDistributed(Flatten())(conv_layer)
    lstm_layer = LSTM(50, activation='relu', return_sequences=False)(flatten_layer)

    # Decoder
    repeat_vector = RepeatVector(input_shape[0])(lstm_layer)  # Повторяем для временных шагов
    lstm_decoder = LSTM(50, activation='relu', return_sequences=True)(repeat_vector) # return_sequences=True
    time_distributed_layer = TimeDistributed(Dense(32, activation='relu'))(lstm_decoder)
    conv_transpose_layer = TimeDistributed(Conv1DTranspose(filters=32, kernel_size=3, activation='relu', padding='same'))(time_distributed_layer)
    output_layer = TimeDistributed(Dense(input_shape[2]))(conv_transpose_layer)  # Выходной слой

    # Autoencoder Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

try:
    # Building model
    autoencoder = build_cnn_lstm_autoencoder((X.shape[1], X.shape[2]))

    # Model Training
    autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)  # Train to reconstruct the entire sequence

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
