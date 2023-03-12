# このプログラムでやっていること 
# - input 
#   - 卸売データ　train.csv, test.csv
#   - 天候データ　weather.csv
# - output
#   - 加工されたデータ
#       - 

import pandas as pd
import numpy as np


train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
weather_df = pd.read_csv('../data/weather.csv')


def get_year_func(x): return int(str(x)[:4])
def get_month_func(x): return int(str(x)[4:6])


# 月ごとの統計データを取得
weather_df['year'] = weather_df['date'].apply(get_year_func)
weather_df['month'] = weather_df['date'].apply(get_month_func)
agg_cols = ['mean_temp', 'max_temp', 'min_temp', 'sum_rain', 'sun_time', 'mean_humid']
gb_df = weather_df.groupby(['area', 'year', 'month'])[agg_cols].agg(['mean', 'max', 'min']).reset_index()

new_cols = []
for col1, col2 in gb_df.columns:
    if col2:
        new_cols.append(col2 + '_' + col1)
    else:
        new_cols.append(col1)
gb_df.columns = new_cols


# train.csv, test.csvの「各地」というエリアを、天候データでは「全国」というエリアとして扱う
# 「全国」のデータは各地域の平均値とする
agg_cols = [i for i in gb_df.columns if i not in ['year', 'month', 'area']]
tmp_df = gb_df.groupby(['year', 'month'])[agg_cols].agg(['mean']).reset_index()

new_cols = []
for col1, col2 in tmp_df.columns:
    new_cols.append(col1)

tmp_df.columns = new_cols
tmp_df['area'] = '全国'
tmp_df = tmp_df[gb_df.columns]


# 都道府県別と全国平均の値を結合
weather_df = pd.concat([gb_df, tmp_df])
# 卸売データと天候データを結合
all_df = pd.concat([train_df, test_df])

# 卸売データのエリアを取得
area_pairs = all_df['area'].unique()  # 重複無くareaを取得
vegetable_areas = set()
for area_pair in area_pairs:
    areas = area_pair.split('_')
    vegetable_areas |= set(areas)

# 天候データのエリアを取得
weather_areas = weather_df['area'].unique()
area_map = {}
update_area_map = {
    '岩手': '盛岡', '宮城': '仙台', '静岡': '浜松', '沖縄': '那覇', '神奈川': '横浜', '愛知': '名古屋', '茨城': '水戸', '北海道': '帯広', '各地': '全国',
    '兵庫': '神戸', '香川': '高松', '埼玉': '熊谷', '国内': '全国', '山梨': '甲府', '栃木': '宇都宮', '群馬': '前橋', '愛媛': '松山'
}
for vegetable_area in vegetable_areas:
    if vegetable_area not in weather_areas and vegetable_area not in update_area_map:
        area_map[vegetable_area] = '全国'  # 外国は全国とする
    else:
        area_map[vegetable_area] = vegetable_area

area_map = {**area_map, **update_area_map}


def join_func(x): return '_'.join([area_map[i] for i in x.split('_')])


all_df['area'] = all_df['area'].apply(join_func)


test_df = all_df.iloc[train_df.shape[0]:]
train_df = all_df.iloc[:train_df.shape[0]]


test_df.to_csv('../data/tutorial_preprocessed_test.csv', index=False)
train_df.to_csv('../data/tutorial_preprocessed_train.csv', index=False)


area_pairs = all_df['area'].unique()
target_cols = [i for i in weather_df.columns if i != 'area']
area_pair_dfs = []

for area_pair in area_pairs:
    areas = area_pair.split('_')
    if len(areas) > 0:
        area = areas[0]
        base_tmp_df = weather_df[weather_df['area'] == area]
        base_tmp_df = base_tmp_df[target_cols].reset_index(drop=True)
        for area in areas[1:]:
            tmp_df = weather_df[weather_df['area'] == area]
            tmp_df = tmp_df[target_cols].reset_index(drop=True)
            base_tmp_df = base_tmp_df.add(tmp_df)
        base_tmp_df /= len(areas)
        base_tmp_df['area'] = area_pair
        area_pair_dfs.append(base_tmp_df)

area_pair_df = pd.concat(area_pair_dfs)

area_pair_df.to_csv('../data/tutorial_preprocessed_weather.csv', index=False)
