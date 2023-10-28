#coding: utf-8

import warnings
warnings.simplefilter('ignore')
import os
import gc
import glob
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm
import pyarrow.orc
from sklearn.preprocessing import LabelEncoder

# 减少内存占用的函数
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# 读取训练集
df_list = []
for filepath in tqdm(glob.glob('data/train_dataset/*/part-*.snappy.orc')):
    dt = filepath.split('/')[2]
    tmp = pd.read_orc(filepath)
    tmp['dt'] = dt
    df_list.append(tmp)
train_data = pd.concat(df_list).reset_index(drop=True)
to_del_cols = ['bid', 'media_id', 'carrier', 'bid_id', 'click_time', 'tracker']
train_data.drop(to_del_cols, axis=1, inplace=True)
del df_list; gc.collect()

# 读取测试集
df_list = []
for filepath in tqdm(glob.glob('data/test_dataset/uid_type/*/part-*.snappy.orc')):
    dt = filepath.split('/')[3]
    tmp = pd.read_orc(filepath)
    tmp['dt'] = dt
    df_list.append(tmp)
test_data = pd.concat(df_list).reset_index(drop=True)
to_del_cols = ['bid', 'media_id', 'carrier', 'bid_id']
test_data.drop(to_del_cols, axis=1, inplace=True)
del df_list; gc.collect()

# labelencoding 转换减少内存占用
train_len = len(train_data)
for col in tqdm(['imp_id', 'adx_app_id', 'adx_app_name', 'adx_app_pkg', 'adx_slot_id']):
    df_col = pd.concat([train_data[col], test_data[col]]).reset_index()
    df_col[col] = df_col[col].astype(str)
    lbe = LabelEncoder()
    df_col[col] = lbe.fit_transform(df_col[col])
    train_data[col] = df_col[:train_len][col].values
    test_data[col] = df_col[train_len:][col].values

# 类型转换减少内存占用
train_data['uid'] = train_data['uid'].astype(int)
test_data['uid'] = test_data['uid'].astype(int)

# 减少内存占用
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)

# 保存处理好的数据
os.makedirs('data_prepared', exist_ok=True)
train_data.to_feather('data_prepared/train_data.feather')
test_data.to_feather('data_prepared/test_data.feather')
