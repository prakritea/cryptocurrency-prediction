# GRU Regularization Experiment

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Activation, Dropout, LeakyReLU
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# === [🧠 Environment Setup] ===
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === [📦 Load Data] ===
file_path = 'data/bitcoin2018_close.h5'
with h5py.File(file_path, 'r') as hf:
    datas = hf['inputs'][:]
    labels = hf['outputs'][:]
    input_times = hf['input_times'][:]
    output_times = hf['output_times'][:]
    original_inputs = hf['original_inputs'][:]
    original_outputs = hf['original_outputs'][:]
    original_datas = hf['original_datas'][:]

# === [🔍 Prepare Data] ===
training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size]
training_labels = labels[:training_size, :, 0]
validation_datas = datas[training_size:]
validation_labels = labels[training_size:, :, 0]
validation_original_outputs = original_outputs[training_size:]
validation_original_inputs = original_inputs[training_size:]
validation_input_times = input_times[training_size:]
validation_output_times = output_times[training_size:]

step_size = datas.shape[1]
nb_features = datas.shape[2]
output_size = labels.shape[1]
batch_size = 8
epochs = 30
units = 50

scaler = MinMaxScaler()

def fit_gru(reg):
    model = Sequential()
    model.add(GRU(units=units, input_shape=(step_size, nb_features),
                  kernel_regularizer=reg, return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size, epochs=epochs, verbose=0)
    return model

def experiment(reg, nb_repeat=5):
    ground_truth = validation_original_outputs[:, :, 0].reshape(-1)
    scaler.fit(original_datas[:, 0].reshape(-1, 1))
    errors = []
    for _ in range(nb_repeat):
        model = fit_gru(reg)
        predicted = model.predict(validation_datas)
        predicted_inverted = scaler.inverse_transform(predicted).reshape(-1)
        mse = mean_squared_error(ground_truth, predicted_inverted)
        errors.append(mse)
    return errors

# Use native Python floats for regularizer strengths
regs = [
    regularizers.l1(0.0),
    regularizers.l1(0.1),
    regularizers.l1(0.01),
    regularizers.l1(0.001),
    regularizers.l1(0.0001),
    regularizers.l2(0.1),
    regularizers.l2(0.01),
    regularizers.l2(0.001),
    regularizers.l2(0.0001)
]

results = pd.DataFrame()
for reg in regs:
    reg_type = type(reg).__name__
    if reg_type == 'L1':
        name = f"l1 {reg.l1:.4f}, l2 0.0000"
    elif reg_type == 'L2':
        name = f"l1 0.0000, l2 {reg.l2:.4f}"
    else:
        name = f"{reg_type}"
    print(f"Training model with: {name}")
    scores = experiment(reg, nb_repeat=5)
    results[name] = scores

results.describe().to_csv('result/gru_regularization_experiment.csv')
print(results.describe())
