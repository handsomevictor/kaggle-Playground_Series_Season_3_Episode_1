import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, data_type: str = 'train'):
        self.data_type = data_type
        self.data = pd.read_csv('train.csv', index_col=0) if data_type == 'train' else pd.read_csv('test.csv',
                                                                                                   index_col=0)

    @staticmethod
    def get_train_test_split(test_size: float = 0.3, random_state: int = 42):
        X = pd.read_csv('train.csv', index_col=0).drop('MedHouseVal', axis=1)
        y = pd.read_csv('train.csv', index_col=0)['MedHouseVal']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data = Data('train')
    x_train, x_test, y_train, y_test = data.get_train_test_split()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

