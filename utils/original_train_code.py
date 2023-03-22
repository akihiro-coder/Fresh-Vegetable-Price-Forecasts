import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier





train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

data = pd.concat([train, test], sort=False)



data['kind_id'] = data['kind'].factorize()[0]


train = data[:len(train)]
test = data[len(train):]
print(train.head())

delete_columns = ['kind', 'mode_price', 'amount']
y_train = train['mode_price'].astype(int)
X_train = train.drop(delete_columns, axis=1)
X_test = test.drop(delete_columns, axis=1)



model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



submission['mode_price'] = list(map(int, y_pred))
submission.to_csv('../logs/randomforest.csv', index=False)
