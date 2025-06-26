import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras.callbacks import CSVLogger, ModelCheckpoint

# ========== 🧠 GPU Configuration ==========
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use '0' for first GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU config error:", e)

# ========== 📂 File Check ==========
file_path = 'data/bitcoin2018_close.h5'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# ========== 📥 Load Dataset ==========
with h5py.File(file_path, 'r') as hf:
    datas = hf['inputs'][:]
    labels = hf['outputs'][:]

print("✅ Dataset loaded.")
print("Input shape:", datas.shape)
print("Label shape:", labels.shape)

# ========== 📌 CNN Model Settings ==========
output_file_name = 'bitcoin_CNN_2layers_relu'
step_size = datas.shape[1]          # Time steps per sample
nb_features = datas.shape[2]        # Features per timestep
batch_size = 8
epochs = 100

# ========== ✂️ Split Train/Validation ==========
training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size, :]
training_labels = labels[:training_size, :]
validation_datas = datas[training_size:, :]
validation_labels = labels[training_size:, :]

# ========== 🧠 Build CNN Model ==========
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=20, strides=3, activation='relu', input_shape=(step_size, nb_features)))
model.add(Dropout(0.5))
model.add(Conv1D(filters=nb_features, kernel_size=16, strides=4))

model.compile(optimizer='adam', loss='mse')
model.summary()

# ========== 🏋️‍♀️ Train the Model ==========
# Ensure weights directory exists
os.makedirs("weights", exist_ok=True)

model.fit(
    training_datas, training_labels,
    validation_data=(validation_datas, validation_labels),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[
        CSVLogger(f"{output_file_name}.csv", append=True),
        ModelCheckpoint(f"weights/{output_file_name}" + "-{epoch:02d}-{val_loss:.5f}.keras",
                monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    ]
)

print("✅ Training completed.")
