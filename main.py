import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten

# === 1. Загрузка данных из CSV файла ===
try:
    # Укажите путь к вашему файлу и используйте правильную кодировку и разделитель
    data = pd.read_csv('path_to_your_file.csv', delimiter=';', encoding='cp1251')
    print("Данные успешно загружены.")
except FileNotFoundError:
    print("Ошибка: Файл не найден. Проверьте путь к файлу.")
    exit()
except pd.errors.EmptyDataError:
    print("Ошибка: Файл пуст.")
    exit()
except Exception as e:
    print(f"Произошла ошибка при загрузке данных: {e}")
    exit()

# === 2. Предобработка данных ===
try:
    # Убираем лишние пробелы из имён колонок
    data.columns = data.columns.str.strip()

    # Проверяем наличие колонки с временными метками и удаляем её
    if 'Время захвата пакетов' in data.columns:
        data.drop(columns=['Время захвата пакетов'], inplace=True)

    # Преобразуем данные в числовой формат
    data = data.apply(pd.to_numeric, errors='coerce')

    # Нормализация данных
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)  # Применяем нормализацию к данным

    print("Данные успешно предобработаны.")
except KeyError as e:
    print(f"Ошибка: Столбец не найден: {e}")
except ValueError as e:
    print(f"Ошибка при преобразовании данных: {e}")
except Exception as e:
    print(f"Произошла ошибка при предобработке данных: {e}")
    exit()


# === 3. Создание временных рядов ===
def create_dataset(data, time_step=1):
    X = []  # Список для хранения входных данных
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), :]  # Получаем временной ряд длиной time_step
        X.append(a)  # Добавляем его в X
    return np.array(X)  # Возвращаем массив X


time_step = 10  # Устанавливаем количество временных шагов для анализа

try:
    X = create_dataset(scaled_data, time_step)  # Создаем набор данных

    # Преобразуем данные в трёхмерный формат для Conv1D
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Убедимся, что данные трёхмерные

    print("Форма входного набора:", X.shape)  # Проверяем форму полученного набора данных
except Exception as e:
    print(f"Ошибка при создании временных рядов: {e}")
    exit()


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
    decoder_layer = TimeDistributed(Dense(input_shape[1]))(lstm_layer)

    # Создание модели автоэнкодера
    model = Model(inputs=input_layer, outputs=decoder_layer)

    model.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с MSE как функцией потерь

    return model


try:
    autoencoder = build_cnn_lstm_autoencoder((X.shape[1], X.shape[2]))  # Создаем автоэнкодер

    # Обучение автоэнкодера на входных данных (X)
    autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

    print("Модель успешно обучена.")
except Exception as e:
    print(f"Ошибка при создании или обучении модели: {e}")
    exit()

# === 5. Обнаружение аномалий ===
try:
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
except Exception as e:
    print(f"Ошибка при обнаружении аномалий: {e}")
