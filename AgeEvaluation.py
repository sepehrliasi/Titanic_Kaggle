import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

train_data = pd.read_csv('train.csv')
age_list = train_data[['PassengerId', 'Survived', 'Age']]
age_list = age_list.dropna(subset=['Age'])
survived = age_list[age_list['Survived'].isin([1])]
died = age_list[age_list['Survived'].isin([0])]
print(age_list.info())
# print('max', max(age_list['Age']))     Max is 80

age_list['bins'] = pd.cut(age_list['Age'], bins=[0, 20, 40, 60, 80, 100],
                          labels=['0-20', '21-40', '41-60',
                                  '61-80', '81-100'])

# print(age_list)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(survived['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], color='green',
         weights=np.ones(len(survived)) / len(survived), alpha=0.7, edgecolor='yellow')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Survived age distribution')

# print(np.ones(len(died)) / len(died))
plt.subplot(1, 2, 1)
plt.hist(died['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], color='red', weights=np.ones(len(died)) / len(died),
         alpha=0.5)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Died age distribution')
plt.title('Died & Survived age distribution')

plt.show()

# plt.hist(age_list['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], weight = age_list['Age']/sum(age_list['Age']))
# plt.show()
