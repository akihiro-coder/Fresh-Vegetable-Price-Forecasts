import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb



train = pd.read_csv('./titanic_data/train.csv')
test = pd.read_csv('./titanic_data/test.csv')
gender_submission = pd.read_csv('./titanic_data/gender_submission.csv')

data = pd.concat([train, test], sort=False)

# 欠損値の補完と文字列の数値化 = 特徴量エンジニアリング
data['Sex'].replace(['male','female'], [0, 1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
data['IsAlone'] = 0 # 'IsAlone'列を追加して初期値を入れておく
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1 # FamilySize==1の箇所だけ1を代入（one-hot）

# その他, 推測に寄与しないだろうパラメータを削除
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


## 説明変数と目的変数
train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived'] # 目的変数　訓練
X_train = train.drop('Survived', axis=1) # 説明変数　訓練
X_test = test.drop('Survived', axis=1) # 評価データ　説明変数


# LightGBMモデルを使用するための下準備
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
categorical_features = ['Embarked', 'Pclass', 'Sex']

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

param = {
    'objective': 'binary',
    'max_bin':300,
    'learning_rate':0.05,
    'num_leaves':40
}

model = lgb.train(
    param, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=10,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(10)]
)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)

## 提出ファイル作成
sub = gender_submission
sub['Survived'] = y_pred
sub.to_csv('./logs/submission_lightgbm_handtuning.csv', index=False)






