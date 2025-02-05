from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === 1. Загрузка данных из CSV файла ===
data = pd.read_csv('data.csv', delimiter=';', encoding='cp1251')

# === 2. Предобработка данных ===
data.columns = data.columns.str.strip()  # Убираем лишние пробелы из имён колонок

if 'Время захвата пакетов' in data.columns:
    data.drop(columns=['Время захвата пакетов'], inplace=True)  # Удаляем колонку с временными метками

# Заменяем запятые на точки для корректного преобразования в числовой формат
data = data.replace(',', '.', regex=True)

# Преобразуем данные в числовой формат
data = data.apply(pd.to_numeric, errors='coerce')  # Преобразуем данные в числовой формат

# Нормализация данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)  # Нормализация данных

# === 3. Создание временных рядов ===
def create_dataset(data, time_step=1):
    X = []  # Список для хранения входных данных
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), :]  # Получаем временной ряд длиной time_step
        X.append(a)  # Добавляем его в X
    return np.array(X)  # Возвращаем массив X


time_step = 10  # Устанавливаем количество временных шагов для анализа
X = create_dataset(scaled_data, time_step)  # Создаем набор данных

# Преобразуем данные в трёхмерный формат для Conv1D
X = np.expand_dims(X, axis=-1)  # Добавляем третье измерение для признаков

print("Форма входного набора:", X.shape)  # Проверяем форму полученного набора данных


# === 4. Создание модели автоэнкодера на основе CNN и LSTM ===
def build_cnn_lstm_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)  # Входной слой с заданной формой

    # Сверточный слой для извлечения пространственных признаков
    conv_layer = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(input_layer)

    # Слой уплощения перед передачей в LSTM (опционально)
    flatten_layer = TimeDistributed(Flatten())(conv_layer)

    # LSTM слой для обработки временной зависимости
    lstm_layer = LSTM(50, activation='relu', return_sequences=True)(flatten_layer)

    # Полносвязный слой декодера для восстановления временных рядов
    decoder_layer = TimeDistributed(Dense(input_shape[2]))(lstm_layer)

    # Создание модели автоэнкодера
    model = Model(inputs=input_layer, outputs=decoder_layer)

    model.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с MSE как функцией потерь

    return model


try:
    autoencoder = build_cnn_lstm_autoencoder((X.shape[1], X.shape[2], X.shape[3]))  # Создаем автоэнкодер

    # Обучение автоэнкодера на входных данных (X)
    autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

    print("Модель успешно обучена.")
except Exception as e:
    print(f"Ошибка при создании или обучении модели: {e}")
    exit()

# === 5. Обнаружение аномалий ===
# Пример вычисления ошибки восстановления
try:
    reconstructed_data = autoencoder.predict(X)

    # Убедимся в том, что выходные данные имеют правильную форму
    if reconstructed_data.shape != X.shape:
        print(f"Форма восстановленных данных: {reconstructed_data.shape}, форма входных данных: {X.shape}")
        reconstructed_data = np.reshape(reconstructed_data, X.shape)  # Приводим к одной форме если необходимо

    # Вычисление ошибки восстановления (MSE) для каждого временного ряда
    mse = np.mean(np.power(X - reconstructed_data, 2), axis=(1, 2))

    # Установка порога для обнаружения аномалий
    threshold = np.mean(mse) + 3 * np.std(mse)

    # Аномалии — это те точки, где ошибка восстановления выше порога
    anomalies = mse > threshold

    # Вывод индексов аномалий
    print("Аномалии обнаружены на индексах:", np.where(anomalies)[0])
except ValueError as e:
    print(f"Ошибка при вычислении аномалий: {e}")
except Exception as e:
    print(f"Произошла ошибка при обнаружении аномалий: {e}")
