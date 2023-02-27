import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier




train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
weather_df = pd.read_csv('../data/weather.csv')
submission = pd.read_csv('../data/sample_submission.csv')


year = lambda x: int(str(x)[:4])
month = lambda x: int(str(x)[4:6])
weather_df['year'] = weather_df['date'].apply(year)
weather_df['month'] = weather_df['date'].apply(month)
agg_cols = ['mean_temp', 'max_temp', 'min_temp', 'sum_rain', 'sun_time', 'mean_humid']
gb_df = weather_df.groupby(['area', 'year', 'month'])[agg_cols].agg(['mean', 'max', 'min']).reset_index()

new_cols = []
for col1, col2 in gb_df.columns:
    if col2:
        new_cols.append(col2 + '_' + col1)
    else:
        new_cols.append(col1)
gb_df.columns = new_cols


agg_cols = [i for i in gb_df.columns if i not in ['year', 'month', 'area']]
tmp_df = gb_df.groupby(['year', 'month'])[agg_cols].agg(['mean']).reset_index()

new_cols = []
for col1, col2 in tmp_df.columns:
    new_cols.append(col1)

tmp_df.columns = new_cols
tmp_df['area'] = '全国'
tmp_df = tmp_df[gb_df.columns]
print(tmp_df.head())

