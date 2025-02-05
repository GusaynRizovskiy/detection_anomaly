import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten

# === 1. Загрузка и предобработка данных ===

# Загрузка данных из CSV файла
data = pd.read_csv('data.csv')  # Укажите путь к вашему файлу

# Нормализация данных
# Позволяет привести данные к одному диапазону и масштабу, что дает возможность
# объединить различные признаки в одной модели без потери информации о различиях между ними
scaler = MinMaxScaler()  # Создаем объект для нормализации
scaled_data = scaler.fit_transform(data)  # Применяем нормализацию к данным


# Преобразование данных в формат временных рядов
def create_dataset(data, time_step=1):
    X = []  # Список для хранения входных данных
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), :]  # Получаем временной ряд длиной time_step
        X.append(a)  # Добавляем его в X
    return np.array(X)  # Возвращаем массив X


time_step = 10  # Количество временных шагов для анализа
X = create_dataset(scaled_data, time_step)  # Создаем набор данных


# === 2. Создание модели автоэнкодера на основе CNN и LSTM ===

def build_cnn_lstm_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)  # Входной слой с заданной формой

    # Сверточный слой для извлечения пространственных признаков
    conv_layer = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(input_layer)

    # Слой уплощения перед передачей в LSTM (опционально)
    flatten_layer = TimeDistributed(Flatten())(conv_layer)

    # LSTM слой для обработки временной зависимости
    lstm_layer = LSTM(50, activation='relu', return_sequences=True)(flatten_layer)

    # Полносвязный слой декодера для восстановления временных рядов
    decoder_layer = TimeDistributed(Dense(input_shape[1]))(lstm_layer)

    # Создание модели автоэнкодера
    model = Model(inputs=input_layer, outputs=decoder_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с MSE как функцией потерь

    return model


# Создаем автоэнкодер с формой входных данных (количество временных шагов и количество признаков)
autoencoder = build_cnn_lstm_autoencoder((X.shape[1], X.shape[2]))

# Обучение автоэнкодера на входных данных (X)
autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

# === 3. Обнаружение аномалий ===

# Получение восстановленных данных от автоэнкодера
reconstructed_data = autoencoder.predict(X)

# Вычисление ошибки восстановления (MSE) для каждого временного ряда
mse = np.mean(np.power(X - reconstructed_data, 2), axis=(1, 2))

# Установка порога для обнаружения аномалий (например, среднее значение + 3 стандартных отклонения)
threshold = np.mean(mse) + 3 * np.std(mse)

# Аномалии — это те точки, где ошибка восстановления выше порога
anomalies = mse > threshold

# Вывод индексов аномалий
print("Аномалии обнаружены на индексах:", np.where(anomalies)[0])
