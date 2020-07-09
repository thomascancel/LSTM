import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.plotting import main_plot_history, main_plot_histogram
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, LSTM


class Optimisation_1:
    """Class which use the TPE algorithm for the surrogate of the objective function"""

    def __init__(self, x_train, y_train, x_val, y_val, epochs, patience, seed_value):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.patience = patience
        self.seed_value = seed_value
        self.space = {
            'batch_size': hp.uniformint('batch_size', 1, 1024),
            'optimizer': hp.choice('optimizer', ['nadam', 'adam', 'sgd'])
        }

    def objective_func(self, params):
        print('Params testing: ', params)
        print('\n')
        model = Sequential()

        model.add(LSTM(units=100, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), return_sequences=True,
                       kernel_initializer=initializers.glorot_uniform(seed=self.seed_value)))
        model.add(Dropout(rate=0.2, seed=self.seed_value))

        model.add(LSTM(units=100, return_sequences=True,
                       kernel_initializer=initializers.glorot_uniform(seed=self.seed_value)))
        model.add(Dropout(rate=0.2, seed=self.seed_value))

        model.add(LSTM(units=100, return_sequences=False,
                       kernel_initializer=initializers.glorot_uniform(seed=self.seed_value)))
        model.add(Dropout(rate=0.2, seed=self.seed_value))

        model.add(
            Dense(units=self.x_train.shape[2], activation='linear',
                  kernel_initializer=initializers.glorot_uniform(seed=self.seed_value)))

        model.compile(loss='mse', optimizer=params['optimizer'])

        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience)]

        result = model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=params['batch_size'], verbose=0,
                           validation_data=(self.x_val, self.y_val), callbacks=callbacks)

        val_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch: ', val_loss)
        return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

    def surrogate(self):
        trials = Trials()
        best = fmin(self.objective_func, self.space, algo=tpe.suggest, max_evals=50, trials=trials)
        hyperparams = space_eval(self.space, best)
        main_plot_history(trials)
        main_plot_histogram(trials)
        return hyperparams


class Optimisation_2:

    def __init__(self, x_train, y_train, x_val, y_val, epochs, patience, seed_value):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.patience = patience
        self.seed_value = seed_value
        self.space = {'choice': hp.choice('num_layers',
                                          [{'layers': 'one',
                                            'units1': hp.uniformint('units1', 10, 500),
                                            'dropout1': hp.uniform('dropout1', .1, .5),
                                            },
                                           {'layers': 'two',
                                            'units2_1': hp.uniformint('units2_1', 10, 500),
                                            'dropout2_1': hp.uniform('dropout2_1', .1, .5),
                                            'units2_2': hp.uniformint('units2_2', 10, 500),
                                            'dropout2_2': hp.uniform('dropout2_2', .1, .5),
                                            },
                                           {'layers': 'three',
                                            'units3_1': hp.uniformint('units3_1', 10, 500),
                                            'dropout3_1': hp.uniform('dropout3_1', .1, .5),
                                            'units3_2': hp.uniformint('units3_2', 10, 500),
                                            'dropout3_2': hp.uniform('dropout3_2', .1, .5),
                                            'units3_3': hp.uniformint('units3_3', 10, 500),
                                            'dropout3_3': hp.uniform('dropout3_3', .1, .5),
                                            }
                                           ]),
                      'batch_size': hp.uniformint('batch_size', 1, 1024),
                      'optimizer': hp.choice('optimizer', ['nadam', 'adam', 'sgd'])
                      }

    def objective_func(self, params):
        print('Params testing: ', params)
        print('\n')
        model = Sequential()

        if params['choice']['layers'] == 'one':
            model.add(LSTM(units=params['choice']['units1'], input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                           return_sequences=False))
            model.add(Dropout(params['choice']['dropout1']))

        if params['choice']['layers'] == 'two':
            model.add(
                LSTM(units=params['choice']['units2_1'], input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                     return_sequences=True))
            model.add(Dropout(params['choice']['dropout2_1']))
            model.add(LSTM(units=params['choice']['units2_2'], return_sequences=False))
            model.add(Dropout(params['choice']['dropout2_2']))

        if params['choice']['layers'] == 'three':
            model.add(
                LSTM(units=params['choice']['units3_1'], input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                     return_sequences=True))
            model.add(Dropout(params['choice']['dropout3_1']))
            model.add(LSTM(units=params['choice']['units3_2'], return_sequences=True))
            model.add(Dropout(params['choice']['dropout3_2']))
            model.add(LSTM(units=params['choice']['units3_3'], return_sequences=False))
            model.add(Dropout(params['choice']['dropout3_3']))

        model.add(Dense(self.x_train.shape[2], activation='linear'))
        model.compile(loss='mse', optimizer=params['optimizer'])

        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience)]

        result = model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=params['batch_size'],
                           verbose=0,
                           validation_data=(self.x_val, self.y_val), callbacks=callbacks)

        val_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch: ', val_loss)
        return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

    def surrogate(self):
        trials = Trials()
        best = fmin(self.objective_func, self.space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('\n')
        print("-" * 80)
        print('best: ', best)
