import pandas as pd

train_data = pd.read_csv('train.csv')
sex_list = train_data[['PassengerId', 'Survived', 'Sex']]
print(sex_list.info())

males = 0
females = 0
male_survivors = 0
female_survivors = 0
num = 0
print(sex_list.shape)
# print(sex_list)
for row in sex_list.itertuples():
    num += 1
    if row[3] == 'male':
        males += 1
        if row[2] == 1:
            male_survivors += 1
    else:
        females += 1
        if row[2] == 1:
            female_survivors += 1

print(num)
print('number of males:', males)
print('number of males survived:', male_survivors)
print('male survival percentage:', male_survivors/males)
print('number of females:', females)
print('number of females survived:', female_survivors)
print('female survival percentage:', female_survivors/females)
