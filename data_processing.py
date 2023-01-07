import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing


# def process_data(df, scaler_type='minmax'):
#     raw_cols = df.columns
#     if scaler_type == 'minmax':
#         # use minmax scaler
#         scaler = MinMaxScaler()
#         # use the original last column
#         return pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=raw_cols[:-1]).join(df.iloc[:, -1])
#
#     elif scaler_type == 'standard':
#         # use standard scaler
#         scaler = StandardScaler()
#         # use the original last column
#         return pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=raw_cols[:-1]).join(df.iloc[:, -1])


class Data:
    def __init__(self, data_type: str = 'train', scaler_type: str = 'minmax', use_original_data=False):
        self.data_type = data_type
        self.data = pd.read_csv('train.csv', index_col=0) if data_type == 'train' else pd.read_csv('test.csv',
                                                                                                   index_col=0)
        self.scaler_type = scaler_type
        self.use_original_data = use_original_data

        if self.data_type == 'train':
            self.processed_data = self.process_data(self.data, scaler_type=scaler_type, handle_last_col=False)
        if self.data_type == 'test':
            self.processed_data = self.process_data(self.data, scaler_type=scaler_type, handle_last_col=True)

        if self.use_original_data:
            self.data = self.add_raw_data_to_train(self.data)

    @staticmethod
    def add_raw_data_to_train(df):
        original_df = fetch_california_housing()['data']
        original_df_train = pd.DataFrame(original_df, columns=fetch_california_housing()['feature_names'])
        original_df_train['MedHouseVal'] = fetch_california_housing()['target']
        return pd.concat([df, original_df_train], axis=0)

    @staticmethod
    def process_data(df, scaler_type='minmax', handle_last_col=True):
        raw_cols = df.columns
        raw_index = df.index
        if scaler_type == 'minmax':
            # use minmax scaler
            scaler = MinMaxScaler()

            if handle_last_col:
                # handle all columns
                return pd.DataFrame(scaler.fit_transform(df), columns=raw_cols, index=raw_index)
            else:
                # use the original last column
                return pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=raw_cols[:-1], index=raw_index).join(
                    df.iloc[:, -1])

        elif scaler_type == 'standard':
            # use standard scaler
            scaler = StandardScaler()
            # use the original last column

            if handle_last_col:
                # handle all columns
                return pd.DataFrame(scaler.fit_transform(df), columns=raw_cols, index=raw_index)
            else:
                return pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=raw_cols[:-1], index=raw_index). \
                    join(df.iloc[:, -1])

        elif scaler_type == 'none':
            return df

    def get_train_test_split(self, test_size: float = 0.3, random_state: int = 42):
        if self.use_original_data:
            raw_train = self.add_raw_data_to_train(pd.read_csv('train.csv', index_col=0))
        else:
            raw_train = pd.read_csv('train.csv', index_col=0)
        processed_train = self.process_data(raw_train, scaler_type=self.scaler_type, handle_last_col=False)

        X = processed_train.drop('MedHouseVal', axis=1)
        y = processed_train['MedHouseVal']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data = Data('train', scaler_type='none', use_original_data=True)
    x_train, x_test, y_train, y_test = data.get_train_test_split()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train.head())

    # df = pd.read_csv('train.csv', index_col=0)
    # print(process_data(df, scaler_type='minmax').head())
