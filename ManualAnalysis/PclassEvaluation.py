import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../train.csv')
class_list = train_data[['PassengerId', 'Survived', 'Pclass']]
class_list = class_list.dropna(subset=['Pclass'])
survived = class_list[class_list['Survived'].isin([1])]
died = class_list[class_list['Survived'].isin([0])]
print(class_list.info())

plt.figure(figsize=(10, 4))

plt.hist(survived['Pclass'], bins=[1, 2, 3, 4], rwidth=0.5, color='green', alpha=0.7, edgecolor='yellow')

plt.hist(died['Pclass'], bins=[1, 2, 3, 4], rwidth=0.6, color='red', alpha=0.5)

plt.title('Died & Survived Pclass distribution')

plt.show()
