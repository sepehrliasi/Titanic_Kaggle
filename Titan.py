import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')
# print(train_data.head())
# print(train_data.shape)
# print(train_data.info())

train_data = train_data.drop('Cabin', axis=1)
# print(train_data.info())

train_data = train_data.dropna()
# print(train_data.info())

