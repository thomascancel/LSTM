import os
import numpy as np
import pandas as pd


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, data, split):
        i_split = int(len(data) * split)
        self.data_train = data.iloc[:i_split].values
        self.data_test = data.iloc[i_split:].values
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def get_test_data(self, seq_len, normalise):
        """Create x, y test data windows"""
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, :]
        return x,y

    def get_predict_data(self, seq_len,  normalise):
        """Create x predict data window"""
        data_windows = [self.data_test[self.len_test-seq_len:]]

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, 1:]
        return x

    def get_train_data(self, seq_len, normalise, validation_split):
        """Create x, y train data windows"""
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        x_train = np.array(data_x[:int(len(data_x)*(1-validation_split))])
        x_val = np.array(data_x[int(len(data_x)*(1-validation_split)):])
        y_train = np.array(data_y[:int(len(data_y)*(1-validation_split))])
        y_val = np.array(data_y[int(len(data_y)*(1-validation_split)):])
        return x_train, y_train, x_val, y_val

    def _next_window(self, i, seq_len, normalise):
        """Generates the next data window from the given index location i"""
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, :]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """Normalise window with a base value of zero"""
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)