import csv

import numpy as np
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("train.csv")

d = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(d)
d = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked'] = df['Embarked'].map(d)

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived']]
df = df.dropna()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

X = df[features]
y = df['Survived']
print(X.info())

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
# plt.show()

test = pandas.read_csv('test.csv')
pass_id = test['PassengerId']

d = {'male': 0, 'female': 1}
test['Sex'] = test['Sex'].map(d)
d = {'C': 0, 'Q': 1, 'S': 2}
test['Embarked'] = test['Embarked'].map(d)

mean_age = np.mean(test['Age'])
test['Age'].fillna(mean_age, inplace=True)
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

print(test.info())

output = dtree.predict(X_test)

results = []
results.append(['PassengerId', 'Survived'])
# print(results)
id = 0
for item in output:
    results.append((pass_id[id], output[id]))
    id += 1

# print(results)
with open('results1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)
