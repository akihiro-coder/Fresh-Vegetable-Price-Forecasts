import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt



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

# 仮説：一緒に乗船した家族の人数が多いほうが、生存率が低そうだ
# 家族人数 = 両親、子供の数 + 兄弟、配偶者の数 + 本人
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
train['FamilySize'] = data['FamilySize'][:len(train)]
test['FamilySize'] = data['FamilySize'][len(train):]

#sns.countplot(x='FamilySize', data=train, hue='Survived')
#plt.savefig('./logs/familySize_Survived.png')
# →　FamilySize >= 5の場合、生存率が低いため、この特徴量が予測精度に寄与しそう
# →　FamilySize == 1の人が圧倒的に多くて、かつ生存率が低い →　FamilySize == 1がかなり予測精度に寄与しそうなため、IsAlone列を追加してみる
data['IsAlone'] = 0 # 'IsAlone'列を追加して初期値を入れておく
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1 # FamilySize==1の箇所だけ1を代入（one-hot）
train['IsAlone'] = data['IsAlone'][:len(train)]
test['IsAlone'] = data['IsAlone'][len(train):]


# 推測に寄与しないだろうパラメータを削除
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


## 説明変数と目的変数
train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived'] # 目的変数　訓練
X_train = train.drop('Survived', axis=1) # 説明変数　訓練
X_test = test.drop('Survived', axis=1) # 評価データ　説明変数



## 機械学習アルゴリズム
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
clf.fit(X_train.drop('IsAlone', axis=1), y_train)
y_pred_familysize = clf.predict(X_test.drop('IsAlone', axis=1))


## 提出ファイル作成
sub = gender_submission
#sub['Survived'] = list(map(int, y_pred))
sub['Survived'] = list(map(int, y_pred_familysize))
sub.to_csv('./logs/submission_familysize.csv', index=False)






