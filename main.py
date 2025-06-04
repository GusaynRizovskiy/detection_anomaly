from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten, RepeatVector
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# === 1. Загрузка и предварительная обработка данных ===
def load_and_preprocess_data(file_path, scaler=None, fit_scaler=False):
    """Загрузка и нормализация данных. Возвращает scaled_data и scaler"""
    data = pd.read_csv(file_path, delimiter=';', encoding='cp1251')
    data.columns = data.columns.str.strip()

    if 'Время захвата пакетов' in data.columns:
        data.drop(columns=['Время захвата пакетов'], inplace=True)

    data = data.replace(',', '.', regex=True)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Нормализация данных
    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)

    return scaled_data, scaler


def load_labels(file_path, time_step):
    """Загрузка меток из файла CIC-IDS2017 и создание меток для окон"""
    labels_data = pd.read_csv(file_path)
    labels = labels_data['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)  # Бинарные метки

    # Создание меток для окон
    window_labels = []
    for i in range(len(labels) - time_step):
        window_label = labels.iloc[i:i + time_step]
        window_labels.append(1 if window_label.any() else 0)

    return np.array(window_labels)


# === 2. Создание временных рядов ===
def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
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


# === 4. Обучение модели на нормальном трафике ===
try:
    # Загрузка и подготовка исходных данных (только нормальный трафик)
    scaled_data, scaler = load_and_preprocess_data('normal_traffic.csv', fit_scaler=True)

    # Создание временных рядов
    time_step = 10
    X_train = create_dataset(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], time_step, scaled_data.shape[1])
    print("Shape of training data:", X_train.shape)

    # Построение и обучение модели
    autoencoder = build_cnn_lstm_autoencoder((X_train.shape[1], X_train.shape[2]))
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

    # Определение порога аномалий
    train_reconstructed = autoencoder.predict(X_train)
    train_mse = np.mean(np.power(X_train - train_reconstructed, 2), axis=(1, 2))
    threshold = np.percentile(train_mse, 99)  # 99-й перцентиль
    np.save('threshold.npy', threshold)
    np.save('scaler.npy', scaler)  # Сохраняем scaler для последующего использования

    print(f"Model trained successfully. Anomaly threshold: {threshold:.4f}")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# === 5. Тестирование на данных с аномалиями ===
try:
    # Загрузка scaler'а и порога
    scaler = np.load('scaler.npy', allow_pickle=True).item()
    threshold = np.load('threshold.npy')

    # Загрузка тестовых данных (с аномалиями)
    test_data, _ = load_and_preprocess_data('test_traffic.csv', scaler=scaler)
    X_test = create_dataset(test_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], time_step, test_data.shape[1])

    # Загрузка меток из отдельного файла CIC-IDS2017
    y_test = load_labels('Tuesday-WorkingHours.pcap_ISCX.csv', time_step)

    # Проверка совпадения размеров
    if len(y_test) != X_test.shape[0]:
        min_length = min(len(y_test), X_test.shape[0])
        y_test = y_test[:min_length]
        X_test = X_test[:min_length]
        print(f"Adjusted shapes to match: X_test {X_test.shape}, y_test {y_test.shape}")

    # Вычисление MSE и предсказание аномалий
    mse_values = []
    for i in range(X_test.shape[0]):
        reconstructed = autoencoder.predict(X_test[i:i + 1], verbose=0)
        mse = np.mean(np.power(X_test[i:i + 1] - reconstructed, 2))
        mse_values.append(mse)

    y_pred = np.array([1 if mse > threshold else 0 for mse in mse_values])

    # Оценка эффективности
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Визуализация результатов
    plt.figure(figsize=(15, 10))

    # График MSE с порогом
    plt.subplot(2, 1, 1)
    plt.plot(mse_values, label='MSE')
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Time windows')
    plt.ylabel('MSE')
    plt.title('Anomaly Detection Results')
    plt.legend()

    # График истинных и предсказанных меток
    plt.subplot(2, 1, 2)
    plt.plot(y_test, 'g-', label='True labels')
    plt.plot(y_pred, 'r--', label='Predicted labels', alpha=0.7)
    plt.xlabel('Time windows')
    plt.ylabel('Label (0=Normal, 1=Anomaly)')
    plt.title('True vs Predicted Anomalies')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Матрица ошибок
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

except Exception as e:
    print(f"Error during testing: {e}")