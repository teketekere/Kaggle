import csv
import datetime
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.cross_validation import KFold
from SkLearnHelper import SklearnHelper


###
# 前処理系のアレコレ
###
def Preprocessing(data):
    data['Embarked'] = PreproEmbarked(data['Embarked'])
    data['Nameband'] = PreproNameband(data['Name'])
    data['Age'] = PreproAge(data['Age'], data['Nameband'])
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1}).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch']
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


def PreproAge(data, filter):
    ran = range(len(data))
    data = [transAge(filter[i]) if data.isnull()[i] else data[i] for i in ran]
    data = pd.Series(data)
    return data


def transAge(cls):
    if(cls == 0):
        ret = 22.0
    elif(cls == 1):
        ret = 36.0
    elif(cls == 2):
        ret = 33.0
    elif(cls == 3):
        ret = 5.0
    elif(cls == 4):
        ret = 45.0
    else:
        ret = 0
    return ret


###
# trainセットのスプリット
# <param>
# tra:訓練用データ
# lab:正解ラベル
# ts:テストサイズの割合
###
def splitTrainSet(tra, lab, ts):
    Xtra, Xte, ytra, yte = train_test_split(tra, lab, test_size=ts)
    return Xtra, Xte, ytra, yte


###
# SKLEARNの学習器インスタンスをゲット
###
# RandomForest
def GetRFInstance(seed=0):
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }
    rf = SklearnHelper(clf=RandomForestClassifier, seed=seed, params=rf_params)
    return rf


# ExtraTrees
def GetETInstance(seed=0):
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=seed, params=et_params)
    return et


def GetRBFSVCInstance(seed=0):
    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'rbf',
        'C': 0.025
        }
    svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)
    return svc


def GetLSVCInstance(seed=0):
    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
        }
    svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)
    return svc


###
# Scoreを表示
###
def showScore(model, test, label, name):
    score = model.score(test, label)
    print(name + ":" + str(score))


###
# Predict and Save result to csv
###
def SaveTitanicPredictToCsv(model, test, name):
    pid = test['PassengerId']
    pred = list()
    for i in range(len(test)):
        pred.append(*model.predict(test.iloc[i:i+1, 1:]))
    output = pd.DataFrame({'Survived': pred})
    output['PassengerId'] = pid
    d = datetime.datetime.now()
    filename = "./result/{0}res_{1:%Y%m%d%H%M%S}.csv".format(name, d)
    output.to_csv(filename, index=False)


###
# Main function
####

# Load training data
train_df = pd.read_csv("./data/train.csv", header=0)
test_df = pd.read_csv("./data/test.csv", header=0)

# 前処理
train_df = Preprocessing(train_df)
test_df = Preprocessing(test_df)
train_y = train_df['Survived']
train_x = train_df.iloc[:, 2:]
tsize = 0.2
tr_x, te_x, tr_y, te_y = splitTrainSet(train_x, train_y, tsize)

# Learn
# Random Forest
rf = GetRFInstance()
rf.fit(tr_x, tr_y)

# Extra Trees
et = GetETInstance()
et.fit(tr_x, tr_y)

# RBFSVC
rbfsvc = GetRBFSVCInstance()
rbfsvc.fit(tr_x, tr_y)

# LSVC
lsvc = GetLSVCInstance()
lsvc.fit(tr_x, tr_y)

# Score
if(tsize != 0.0):
    showScore(rf, te_x, te_y, "RF")
    showScore(et, te_x, te_y, "ET")
    showScore(rbfsvc, te_x, te_y, "RBFSVC")
    showScore(lsvc, te_x, te_y, "LSVC")

'''
# Predict
SaveTitanicPredictToCsv(rf, test_df, "RF")
SaveTitanicPredictToCsv(et, test_df, "ET")
SaveTitanicPredictToCsv(rbfsvc, test_df, "RSVC")
SaveTitanicPredictToCsv(lsvc, test_df, "LSVC")
'''
