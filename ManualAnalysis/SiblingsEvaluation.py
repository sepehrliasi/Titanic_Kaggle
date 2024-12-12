import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../train.csv')
sib_list = train_data[['PassengerId', 'Survived', 'SibSp']]
sib_list = sib_list.dropna(subset=['SibSp'])
survived = sib_list[sib_list['Survived'].isin([1])]
died = sib_list[sib_list['Survived'].isin([0])]
print(sib_list.info())
# print('max', max(sib_list['SibSp']))     # Max is 8

plt.figure(figsize=(10, 4))

plt.hist(survived['SibSp'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Survived', rwidth=0.55, color='green', alpha=0.7, edgecolor='yellow')

plt.hist(died['SibSp'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Died', rwidth=0.6, color='red', alpha=0.5)

plt.title('Died & Survived Sibling/Spouse distribution')
plt.legend()

plt.show()
