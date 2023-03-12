from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
sns.set(font="IPAexGothic")
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression




train_df = pd.read_csv('../data/tutorial_preprocessed_train.csv')
test_df = pd.read_csv('../data/tutorial_preprocessed_test.csv')
weather_df = pd.read_csv('../data/tutorial_preprocessed_weather.csv')


