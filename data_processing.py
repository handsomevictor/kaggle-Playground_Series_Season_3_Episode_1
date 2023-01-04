import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, data_type: str):
        self.data_type = data_type
        self.data = pd.read_csv('train.csv') if data_type == 'train' else pd.read_csv('test.csv')


if __name__ == '__main__':
    print(Data('train').data)

