import csv
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def Preprocessing(data):
    data['Embarked'] = PreproEmbarked(data['Embarked'])
    data['Nameband'] = PreproNameband(data['Name'])
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1}).astype(int)
    data = data.drop('Name', axis=1)
    data = data.drop('Ticket', axis=1)
    data = data.drop('Cabin', axis=1)
    return data


def PreproEmbarked(data):
    data = data.fillna('S')
    data = data.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return data


def PreproNameband(data):
    nameband = data.str.extract('([A-Za-z]+)\.')
    with open('./config/namebindBefore.csv', 'r') as f:
        reader = csv.reader(f)
        before = list(*reader)
    with open('./config/namebindAfter.csv', 'r') as f:
        reader = csv.reader(f)
        after = list(*reader)
        after = list(map(int, after))
    nameband = nameband.replace(before, after)
    return nameband


# Main
# Load training data
train_df = pd.read_csv("./data/train.csv", header=0)
test_df = pd.read_csv("./data/test.csv", header=0)
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())

# 前処理
train_df = Preprocessing(train_df)
test_df = Preprocessing(test_df)
# print(train_df.head())
# print(test_df.head())

# Learn
model = RandomForestClassifier()
model.fit(train_df.iloc[:, 2:], train_df['Survived'])

# Predict
pid = test_df['PassengerId']
pred = list()
for i in range(len(test_df)):
    pred.append(*model.predict(test_df.iloc[i:i+1, 1:]))

output = pd.DataFrame({'PassengerId': pid,
                      'Survived': pred})
d = datetime.datetime.now()
output.to_csv("./result/predict_{0:%Y%m%d%H%M%S}.csv".format(d), index=False)
