import pandas

df = pandas.read_csv("train.csv")

print(df.info())

df = df[['Survived', 'Cabin']]

survived_cabin = 0
dead_cabin = 0
s = 0
d = 0
print(df)
for item in df.itertuples():
    if pandas.isnull(item[2]):
        if item[1] == 1:
            survived_cabin += 1
        else:
            dead_cabin += 1
    else:
        if item[1] == 1:
            s += 1
        else:
            d += 1

print('survived cabin:', survived_cabin / (survived_cabin+dead_cabin))
print('survived regular:', s / (s+d))

print('finish')