import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
par_list = train_data[['PassengerId', 'Survived', 'Parch']]
par_list = par_list.dropna(subset=['Parch'])
survived = par_list[par_list['Survived'].isin([1])]
died = par_list[par_list['Survived'].isin([0])]
print(par_list.info())
print('max', max(par_list['Parch']))     # Max is 8

plt.figure(figsize=(10, 4))

plt.hist(survived['Parch'], bins=[0, 1, 2, 3, 4, 5, 6, 7], label='Survived', rwidth=0.55, color='green', alpha=0.7, edgecolor='yellow')

plt.hist(died['Parch'], bins=[0, 1, 2, 3, 4, 5, 6, 7], label='Died', rwidth=0.6, color='red', alpha=0.5)

plt.title('Died & Survived Parent/Child distribution')
plt.legend()

plt.show()
