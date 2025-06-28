import os
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU, Activation
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras import regularizers
import tensorflow as tf

# ✅ GPU Setup
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to '0' or '1' depending on available GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU setup failed:", e)

# ✅ Load Data
with h5py.File('data/bitcoin2018_close.h5', 'r') as hf:
    datas = hf['inputs'][:]        # Shape: (samples, time_steps, features)
    labels = hf['outputs'][:]      # Shape: (samples, output_steps, 1)

# ✅ Training Configs
step_size = datas.shape[1]
nb_features = datas.shape[2]
output_size = labels.shape[1]
units = 50
batch_size = 8
epochs = 50
reg = 1e-5  # L1 regularization
output_file_name = 'bitcoin2018_LSTM_tanh_relu'

# ✅ Split into Training & Validation
training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size, :]
training_labels = labels[:training_size, :, 0]
validation_datas = datas[training_size:, :]
validation_labels = labels[training_size:, :, 0]

# ✅ Build Model
model = Sequential()
model.add(LSTM(units=units, 
               activation='tanh',
               activity_regularizer=regularizers.l1(reg),
               input_shape=(step_size, nb_features),
               return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_size))
model.add(LeakyReLU())
model.compile(loss='mse', optimizer='adam')

# ✅ Callbacks
os.makedirs('weights', exist_ok=True)
os.makedirs('result', exist_ok=True)

csv_logger = CSVLogger(f'result/{output_file_name}.csv', append=True)
checkpoint = ModelCheckpoint(
    filepath=f'weights/{output_file_name}-{{epoch:02d}}-{{val_loss:.5f}}.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ✅ Train Model
model.fit(training_datas,
          training_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(validation_datas, validation_labels),
          callbacks=[csv_logger, checkpoint, early_stop])
