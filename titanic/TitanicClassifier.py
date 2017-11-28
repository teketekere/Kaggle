import csv
import datetime
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1}).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
    cabMap = {'A': 1, 'B': 2, 'C': 2, 'D': 2, 'E': 2, 'F': 1, 'G': 1, 'T': 1}
    cabMap.update({'n': 0})
    data['Cabin'] = data['Cabin'].map(cabMap).astype(int)
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 0, 'IsAlone'] = 1
    data['Ticket'] = data['Ticket'].apply(lambda x: len(x))
    # data = data.drop('Name', axis=1)
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


def PreproFamilyName(tr, te):
    data = pd.concat([tr, te], ignore_index=True)
    data = data.apply(lambda x: str(x).split()[0])
    datan = data
    famnameClass = {}
    famnameClass[data[0]] = 0
    for i in range(len(data)):
        if(data[i] in famnameClass.keys()):
            datan[i] = famnameClass[data[i]]
        else:
            famnameClass[data[i]] = max(famnameClass.values()) + 1
            datan[i] = famnameClass[data[i]]
    r1 = data[0:len(tr)]
    r2 = data[len(tr):]
    return [r1, r2.reset_index()]


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
        'verbose': 0}
    rf = SklearnHelper(clf=RandomForestClassifier, seed=seed, params=rf_params)
    return rf


# ExtraTrees
def GetETInstance(seed=0):
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0}
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=seed, params=et_params)
    return et


def GetRBFSVCInstance(seed=0):
    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'rbf',
        'C': 1,
        'gamma': 0.01}
    svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)
    return svc


def GetLSVCInstance(seed=0):
    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.1,
        'gamma': 0.01}
    svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)
    return svc


def GetAdaInstance(seed=0):
    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75}
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=seed, params=ada_params)
    return ada


def GetGBInstance(seed=0):
    # Gradient Boosting parameters
    gb_p = {
        'n_estimators': 500,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0}
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=seed, params=gb_p)
    return gb


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
    pred = model.predict(test.iloc[:, 1:])
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
trn = train_df['Name']
ten = test_df['Name']
train_df['Name'], test_df['Name'] = PreproFamilyName(trn, ten)
train_y = train_df['Survived']
train_x = train_df.iloc[:, 2:]
tsize = 0.0
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

# AdaBoost
ada = GetAdaInstance()
ada.fit(tr_x, tr_y)

# GradientBoosting
gb = GetGBInstance()
gb.fit(tr_x, tr_y)

# Score
if(tsize != 0.0):
    showScore(rf, te_x, te_y, "RF")
    showScore(et, te_x, te_y, "ET")
    showScore(rbfsvc, te_x, te_y, "RBFSVC")
    showScore(lsvc, te_x, te_y, "LSVC")
    showScore(ada, te_x, te_y, "ADA")
    showScore(gb, te_x, te_y, "GB")
else:
    # Predict
    SaveTitanicPredictToCsv(rf, test_df, "RF")
    SaveTitanicPredictToCsv(et, test_df, "ET")
    # SaveTitanicPredictToCsv(rbfsvc, test_df, "RSVC")
    # SaveTitanicPredictToCsv(lsvc, test_df, "LSVC")
    SaveTitanicPredictToCsv(ada, test_df, "Ada")
    SaveTitanicPredictToCsv(gb, test_df, "GB")
