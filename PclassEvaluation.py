import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

train_data = pd.read_csv('train.csv')
class_list = train_data[['PassengerId', 'Survived', 'Pclass']]
class_list = class_list.dropna(subset=['Pclass'])
survived = class_list[class_list['Survived'].isin([1])]
died = class_list[class_list['Survived'].isin([0])]
print(class_list.info())

# print(age_list)
plt.figure(figsize=(10, 4))

plt.hist(survived['Pclass'], color='green', weights=np.ones(len(survived)) / len(survived), alpha=0.7, edgecolor = 'yellow')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.hist(died['Pclass'], color='red', weights=np.ones(len(died)) / len(died), alpha=0.5)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.title('Died & Survived Pclass distribution')

plt.show()