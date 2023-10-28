#coding: utf-8

import warnings
warnings.simplefilter('ignore')
import os
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

# 加载处理过的数据
train_data = pd.read_feather('data_prepared/train_data.feather')
test_data = pd.read_feather('data_prepared/test_data.feather')

submit = pd.DataFrame()
for d in ['2023-05-14', '2023-05-15', '2023-05-16']:
    tmp = pd.read_csv(f'data/submit/submission_{d}.csv')
    tmp['dt'] = d
    submit = pd.concat([submit, tmp])
submit = submit.reset_index(drop=True)

# 测试集有 110 个 image + 大量的 na (na 可以用来做统计特征, 影响 uid 侧特征, 预测时不需要)
# 训练集用前 14 天(负采样), 验证集用最后一天(2023-05-13)的 test_images
test_images = submit['creative_image_md5'].unique().tolist()
print(len(test_images))

# 文本清洗
def text_clean(text):
    clean_text = text.upper()
    clean_text = clean_text.replace(' ', '').replace('_', '')
    return clean_text

for col in tqdm(['make', 'model']):
    train_data[col] = train_data[col].apply(text_clean)
    test_data[col] = test_data[col].apply(text_clean)

# labelcoding
train_len = len(train_data)
for col in tqdm(['make', 'model', 'osv']):
    df_col = pd.concat([train_data[col], test_data[col]]).reset_index()
    df_col[col] = df_col[col].astype(str)
    lbe = LabelEncoder()
    df_col[col] = lbe.fit_transform(df_col[col])
    train_data[col] = df_col[:train_len][col].values
    test_data[col] = df_col[train_len:][col].values

# 统计特征
for col in tqdm(['creative_image_md5']):
    train_data['image_dt_count'] =\
        train_data.groupby([col, 'dt'])[col].transform('count')
    test_data['image_dt_count'] =\
        test_data.groupby([col, 'dt'])[col].transform('count')
    
for col in tqdm(['plan_id', 'campaign_id', 'imp_id', 'adx_app_id', 'adx_app_name',
                 'adx_app_pkg', 'adx_slot_id', 'agent_id', 'net_type', 'region_id',
                 'make', 'model', 'os', 'osv', 'uid']):
    train_data[f'image_{col}_dt_nunique'] =\
        train_data.groupby(['creative_image_md5', 'dt'])[col].transform('nunique')
    test_data[f'image_{col}_dt_nunique'] =\
        test_data.groupby(['creative_image_md5', 'dt'])[col].transform('nunique')

for col in tqdm(['bid_floor']):
    for m in ['max', 'min', 'mean', 'std']:
        train_data[f'image_{col}_dt_{m}'] =\
            train_data.groupby(['creative_image_md5', 'dt'])[col].transform(m) 
        test_data[f'image_{col}_dt_{m}'] =\
            test_data.groupby(['creative_image_md5', 'dt'])[col].transform(m)
    
    
for col in tqdm(['net_type', 'region_id', 'make', 'model', 'os', 'osv']):
    train_data[f'{col}_dt_nunique'] = train_data.groupby(['dt'])[col].transform('nunique')
    test_data[f'{col}_dt_nunique'] = test_data.groupby(['dt'])[col].transform('nunique')
    train_data[f'uid_{col}_dt_nunique'] = train_data.groupby(['uid', 'dt'])[col].transform('nunique')
    test_data[f'uid_{col}_dt_nunique'] = test_data.groupby(['uid', 'dt'])[col].transform('nunique')

# 时间特征
train_data['bid_dt'] = pd.to_datetime(train_data['bid_time'], unit='s')
test_data['bid_dt'] = pd.to_datetime(test_data['bid_time'], unit='s')

train_data['bid_hour'] = train_data['bid_dt'].dt.hour
test_data['bid_hour'] = test_data['bid_dt'].dt.hour
train_data['bid_minute'] = train_data['bid_dt'].dt.minute
test_data['bid_minute'] = test_data['bid_dt'].dt.minute
train_data['bid_second'] = train_data['bid_dt'].dt.second
test_data['bid_second'] = test_data['bid_dt'].dt.second

train_data['bid_dt_seconds'] =\
    3600*train_data['bid_hour'] + 60*train_data['bid_minute'] + train_data['bid_second']
test_data['bid_dt_seconds'] =\
    3600*test_data['bid_hour'] + 60*test_data['bid_minute'] + test_data['bid_second']

for col in tqdm(['bid_dt_seconds']):
    for m in ['max', 'min', 'std']:
        train_data[f'image_{col}_dt_{m}'] =\
            train_data.groupby(['creative_image_md5', 'dt'])[col].transform(m) 
        test_data[f'image_{col}_dt_{m}'] =\
            test_data.groupby(['creative_image_md5', 'dt'])[col].transform(m)

train_data['image_bid_dt_seconds_dt_span'] =\
    train_data['image_bid_dt_seconds_dt_max'] - train_data['image_bid_dt_seconds_dt_min']
test_data['image_bid_dt_seconds_dt_span'] =\
    test_data['image_bid_dt_seconds_dt_max'] - test_data['image_bid_dt_seconds_dt_min']

train_data.drop(['bid_time', 'bid_dt', 'bid_minute', 'bid_second',
                 'image_bid_dt_seconds_dt_min', 'image_bid_dt_seconds_dt_max'],
                axis=1, inplace=True)
test_data.drop(['bid_time', 'bid_dt', 'bid_minute', 'bid_second',
                'image_bid_dt_seconds_dt_min', 'image_bid_dt_seconds_dt_max'],
               axis=1, inplace=True)

# 没太多用处
to_drop_cols = ['creative_texts', 'ip']
train_data.drop(to_drop_cols, axis=1, inplace=True)
test_data.drop(to_drop_cols, axis=1, inplace=True)

# 保存
os.makedirs('features', exist_ok=True)
train_data.to_feather('features/train_feats.feather')
test_data.to_feather('features/test_feats.feather')
