from keras import Sequential
from keras.src.layers import Dense, ReLU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM
import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("train.csv")

# Data clearance
d = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(d)
d = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked'] = df['Embarked'].map(d)
df['Age'].fillna(-10, inplace=True)

df = df[['Pclass', 'Fare', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived']]
df = df.dropna()

features = ['Pclass', 'Fare', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.iloc[:, :-1])  # Exclude target column
scaled_target = df['Survived'].values

X = scaled_features
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = scaled_target
# print(X.info())
print(X.shape)

model = Sequential()
# model.add(LSTM(8, input_shape=(1, 6)))
model.add(LSTM(10, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='linear'))
model.add(ReLU(max_value=1))
model.compile(loss='mse', optimizer ='adam', metrics=['accuracy'])
model.fit(X, y, epochs=120, batch_size=10)
scores = model.evaluate(X, y, verbose=1, batch_size=5)
print('Accuracy: {}'.format(scores[1]))

test = pandas.read_csv('test.csv')
pass_id = test['PassengerId']

d = {'male': 0, 'female': 1}
test['Sex'] = test['Sex'].map(d)
d = {'C': 0, 'Q': 1, 'S': 2}
test['Embarked'] = test['Embarked'].map(d)

mean_age = np.mean(test['Age'])
test['Age'].fillna(mean_age, inplace=True)
X_test = test[['Pclass', 'Fare', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

scaled_test = scaler.fit_transform(X_test.iloc[:, :])
scaled_test = scaled_test.reshape((scaled_test.shape[0], 1, scaled_test.shape[1]))

predict = model.predict(scaled_test)
# print(predict)

# print(test.info())

results = []
results.append(['PassengerId', 'Survived'])
# print(results)
id = 0
for item in predict:
    if predict[id] < 0.5:
        results.append((pass_id[id], 0))
    else:
        results.append((pass_id[id], 1))
    id += 1

# print(results)
with open('resultsLSTM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)
