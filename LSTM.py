import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ========== ⚙️ GPU CONFIG ========== #
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU config error:", e)

# ========== 📥 Load Binance HDF5 Dataset ========== #
with h5py.File('data/bitcoin2018_close.h5', 'r') as hf:
    datas = hf['inputs'][:]                  # (samples, 256, 1)
    labels = hf['outputs'][:]                # (samples, 16, 1)
    input_times = hf['input_times'][:]
    output_times = hf['output_times'][:]
    original_datas = hf['original_datas'][:]
    original_outputs = hf['original_outputs'][:]

# ========== 📊 Parameters ========== #
step_size = datas.shape[1]       # 256
nb_features = datas.shape[2]     # 1
output_size = labels.shape[1]    # 16

# ========== ✂️ Split Dataset ========== #
split = int(0.8 * datas.shape[0])
X_train, X_val = datas[:split], datas[split:]
y_train, y_val = labels[:split, :, 0], labels[split:, :, 0]
input_times_val = input_times[split:]
output_times_val = output_times[split:]
original_outputs_val = original_outputs[split:]

# ========== 🧠 Build LSTM Model ========== #
model = Sequential()
model.add(LSTM(units=50, input_shape=(step_size, nb_features), return_sequences=False))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(output_size))
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='adam')
model.summary()

# ========== 🏋️ Train LSTM ========== #
output_file_name = 'bitcoin2018_LSTM_tanh_relu'
os.makedirs('weights', exist_ok=True)

model.fit(
    X_train, y_train,
    batch_size=8,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[
        CSVLogger(output_file_name + '.csv', append=True),
        ModelCheckpoint(
            f'weights/{output_file_name}-{{epoch:02d}}-{{val_loss:.5f}}.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
)

# ========== 🔮 Predict and Inverse Scale ========== #
predicted = model.predict(X_val)

scaler = MinMaxScaler()
scaler.fit(original_datas[:, 0].reshape(-1, 1))
predicted_inv = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(-1)
actual_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)

# ========== 🕒 Reconstruct Ground Truth & Timestamps ========== #
ground_true = np.append(X_val[:, :, 0], original_outputs_val[:, :, 0], axis=1).reshape(-1)
ground_true_times = np.append(input_times_val, output_times_val, axis=1).reshape(-1)

ground_true_times = pd.to_datetime(ground_true_times, unit='s')
prediction_times = pd.to_datetime(output_times_val.reshape(-1), unit='s')

# ========== 📈 Plotting ========== #
ground_true_df = pd.DataFrame({'times': ground_true_times, 'value': ground_true})
prediction_df = pd.DataFrame({'times': prediction_times, 'value': predicted_inv})

plt.figure(figsize=(20, 10))
plt.plot(ground_true_df.times, ground_true_df.value, label='Actual')
plt.plot(prediction_df.times, prediction_df.value, 'ro', markersize=2, label='Predicted')
plt.title('Bitcoin Price Prediction (Binance 2018 LSTM)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.grid(True)
os.makedirs('result', exist_ok=True)
plt.savefig('result/bitcoin2018_lstm_result.png')
plt.show()

# ========== 📊 MSE Evaluation ========== #
mse = mean_squared_error(y_val.reshape(-1), predicted.reshape(-1))
print("✅ LSTM Model MSE (Binance 2018):", mse)
