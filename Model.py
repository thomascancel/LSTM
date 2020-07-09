import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from numpy import newaxis
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, seed_value):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(units=neurons, activation=activation,
                                     kernel_initializer=initializers.glorot_uniform(seed=seed_value)))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(units=neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,
                                    kernel_initializer=initializers.glorot_uniform(seed=seed_value)))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate=dropout_rate, seed=seed_value))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, patience, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, 'model_momentum_forecast.h5')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

        # Plot the loss function
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.plot(history.history['loss'], 'r', label='train')
        ax.plot(history.history['val_loss'], 'b', label='val')
        ax.set_xlabel(r'Epoch', fontsize=20)
        ax.set_ylabel(r'Loss', fontsize=20)
        ax.legend()
        ax.tick_params(labelsize=20)
        plt.savefig(os.path.join(save_dir, 'Loss.pdf'))

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 'prediction_len' steps before shifting prediction run forward by 'prediction_len' steps
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :]))
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def forecast_sequences_multiple(self, data, window_size, prediction_len):
        print('[Model] Forecasting Sequences Multiple...\n')
        prediction_seqs = []
        curr_frame = data
        predicted = []
        for j in range(prediction_len):
            predicted.append(self.model.predict(curr_frame))
            curr_frame = curr_frame[:, 1:, :]
            curr_frame = np.insert(curr_frame[0], window_size - 2, predicted[-1], axis=0)
            curr_frame = curr_frame[newaxis, :, :]
        prediction_seqs.append(predicted)
        return prediction_seqs[0]
