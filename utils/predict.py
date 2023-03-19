from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
sns.set(font="IPAexGothic")


train_df = pd.read_csv('../data/tutorial_preprocessed_train.csv')
test_df = pd.read_csv('../data/tutorial_preprocessed_test.csv')
weather_df = pd.read_csv('../data/tutorial_preprocessed_weather.csv')

# (77751, 7)
# (315, 7)
# (46872, 21)
# print(test_df.shape)
# print(weather_df.shape)

train_df['year'] = train_df['date'] // 10000
test_df['year'] = test_df['date'] // 10000
train_df['month'] = train_df['date'].apply(lambda x: int(str(x)[4:6]))
test_df['month'] = test_df['date'].apply(lambda x: int(str(x)[4:6]))
# print(train_df.head())


TARGET = 'mode_price'
price = train_df.groupby('kind')[[TARGET]].mean().sort_values(by=TARGET, ascending=False)


kinds = test_df['kind'].unique()
train_df = train_df[train_df['kind'].isin(kinds)]
table = pd.pivot_table(train_df.query('20211101 <= date <= 20221031'), index='kind', columns='month', values=TARGET, aggfunc='count')


all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
all_df.drop('weekno', axis=1, inplace=True)

# 取引の無い日のレコード分のダミーデータを作成　綺麗に可視化されるようにする
max_days = (datetime(2022, 12, 31) - datetime(2005, 1, 1)).days
dum_data = []
for i in range(max_days + 1):
    date = datetime(2005, 1, 1) + timedelta(days=i)
    y, wn = date.isocalendar()[0], date.isocalendar()[1]
    date = int(date.strftime('%Y%m%d'))
    m = int(str(date)[4:6])
    dum_data.append(['ダミー', date, 0, 0, 'ダミー', y, m])

dum_df = pd.DataFrame(dum_data, columns=all_df.columns)

vis_df = pd.concat([all_df, dum_df])
vis_df = vis_df.query('20161101 <= date <= 20221031').reset_index(drop=True)
vis_df = pd.pivot_table(vis_df, index='date', columns='kind', values='mode_price').reset_index()
vis_df.fillna(0, inplace=True)
print(vis_df.head())


nrow = 4
ncol = 4

fig, ax = plt.subplots(nrow, ncol, figsize=(16, 12))
for i, kind in enumerate(kinds):
    if i < nrow * ncol:
        df = vis_df.loc[:, ['date', kind]]
        df.columns = ['date', TARGET]
        j = i // ncol
        k = i % ncol
        ax[j, k].plot(df['date'].to_list(), df[TARGET].to_list())
        ax[j, k].set_title(kind)
plt.tight_layout()
plt.show()
plt.close()


# 各年の価格の先行指標となる変数を探す必要がある
