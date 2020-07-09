import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import Timer
from Extract_data import Extraction
from Data_processor import DataLoader
from Model import Model
from Bayesian_opt import Optimisation_1, Optimisation_2

# Fix seed value to get reproducible results
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
tf.random.set_seed(seed_value)
from tensorflow.python.keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


def plot_results_multiple(predicted_data, true_data, prediction_len, col, save_test):
    for j in range(true_data.shape[1]):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data[:, j], label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            data_i = [item[0][j] for item in data]
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data_i)
            plt.legend()
        plt.savefig(os.path.join(save_test, 'Momentum trend.png'), bbox_inches='tight', dpi=400)
        plt.close()


def download_data():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['data']['save_dir']): os.makedirs(configs['data']['save_dir'])
    Extraction(configs['data']['ticker'], configs['data']['column'], configs['data']['save_dir']).get_data()


def tune_model(method):
    timer = Timer()
    timer.start()
    configs = json.load(open('config.json', 'r'))
    input = pd.read_csv(os.path.join(configs['data']['save_dir'], 'data.csv'), index_col=0)
    data = DataLoader(input, configs['data']['train_test_split'])

    x_train, y_train, x_val, y_val = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        validation_split=configs['training']['validation_split']
    )

    print("[Bayesian_opt] Starting optimization of the hyperparameters...")

    if method == 'Optimisation_1':
        hyperparams = Optimisation_1(x_train, y_train, x_val, y_val,
                                            epochs=configs['training']['epochs'],
                                            patience=configs['training']['patience'],
                                            seed_value=seed_value).surrogate()

    elif method == 'Optimisation_2':
        hyperparams = Optimisation_2(x_train, y_train, x_val, y_val,
                                            epochs=configs['training']['epochs'],
                                            patience=configs['training']['patience'],
                                            seed_value=seed_value).surrogate()

    else:
        print("Error no methods have the name " + method)

    '''
    print("-" * 80)
    print("Bayesian optimization completed.")
    timer.stop()
    print('best: ', hyperparams)

    configs['training']['batch_size'] = hyperparams["batch_size"]
    configs['model']['optimizer'] = hyperparams["optimizer"]

    with open('config.json', 'w') as jsonFile:
        json.dump(configs, jsonFile, indent=4)

    print("Json file updated")
    print("Model ready to be trained...")
    print("-" * 80)
    '''

def train_model():
    timer = Timer()
    timer.start()
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    if not os.path.exists(configs['model']['save_test']): os.makedirs(configs['model']['save_test'])

    input = pd.read_csv(os.path.join(configs['data']['save_dir'], 'data.csv'), index_col=0)
    data = DataLoader(input, configs['data']['train_test_split'])

    model = Model()
    model.build_model(configs, seed_value)
    x_train, y_train, x_val, y_val = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        validation_split=configs['training']['validation_split']
    )

    # in-memory training
    model.train(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        patience=configs['training']['patience'],
        save_dir=configs['model']['save_dir'],
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    print('[Model] Predicting Sequences Multiple x_test...')

    predictions_test = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
                                                        configs['data']['sequence_length'])

    plot_results_multiple(predictions_test, y_test, configs['data']['sequence_length'],
                          configs['data']['column'], configs['model']['save_test'])

    print('\n')
    print('Plotting Completed. Find the results in the saved_tests directory.\n')
    timer.stop()


def momentum_calc(prediction_data, cols):
    moms = []
    for i in range(len(cols)):
        moms.append(prediction_data[-1][0][i] - prediction_data[0][0][i])
    momentums = pd.DataFrame(moms, index=cols, columns=['Momentums'])
    return momentums


def momentum_forecast():
    configs = json.load(open('config.json', 'r'))
    # if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(configs['data']['train_test_split'])

    model = Model()
    model.load_model(os.path.join('saved_models/model_momentum_forecast.h5'))
    x = data.get_predict_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.forecast_sequences_multiple(x, configs['data']['sequence_length'],
                                                    configs['data']['sequence_length'])

    momentum_pred = momentum_calc(predictions, configs['data']['columns'])

    print("-" * 80)
    for i in range(momentum_pred.shape[0]):
        print('The 30-day momentum of ' + configs['data']['columns'][i] + ' is %s.\n' % momentum_pred['Momentums'][
            configs['data']['columns'][i]])
    print("-" * 80)


if __name__ == '__main__':
    #download_data()
    #tune_model('Optimisation_3')
    train_model()
    #momentum_forecast()
