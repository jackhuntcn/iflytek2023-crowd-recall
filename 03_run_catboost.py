#coding: utf-8

import warnings
warnings.simplefilter('ignore')
import os
import gc
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('float_format', lambda x: '%.4f' % x)
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score
import catboost as cb

# 加载生成好的特征文件
train_data = pd.read_feather('features/train_feats.feather')
test_data = pd.read_feather('features/test_feats.feather')

# 测试集 image 列表
submit = pd.DataFrame()
for d in ['2023-05-14', '2023-05-15', '2023-05-16']:
    tmp = pd.read_csv(f'data/submit/submission_{d}.csv')
    tmp['dt'] = d
    submit = pd.concat([submit, tmp])
submit = submit.reset_index(drop=True)
test_images = submit['creative_image_md5'].unique().tolist()
print(len(test_images))

# 数据划分
# 训练集用前 14 天(负采样), 验证集用最后一天(2023-05-13)的 test_images
df_train = train_data[train_data['dt'] != '2023-05-13'].reset_index(drop=True)
df_valid = train_data[train_data['dt'] == '2023-05-13'].reset_index(drop=True)
del train_data, submit; gc.collect() 
df_train_pos = df_train[df_train['click'] != 0].reset_index(drop=True)
df_train_neg = df_train[df_train['click'] == 0].reset_index(drop=True)
df_train_neg = df_train_neg.sample(n=3900000, random_state=42).reset_index(drop=True)
df_train = pd.concat([df_train_pos, df_train_neg]).sort_values('creative_image_md5').reset_index(drop=True)
df_valid = df_valid[df_valid['creative_image_md5'].isin(test_images)].reset_index(drop=True)
features = [col for col in test_data.columns if col not in ['creative_image_md5', 'uid', 'dt']]
X_train, y_train = df_train[features], df_train['click']
X_valid, y_valid = df_valid[features], df_valid['click']

# 训练及验证
model = cb.CatBoostClassifier(iterations=10000,
                              learning_rate=0.01,
                              depth=6,
                              eval_metric='AUC',
                              task_type='GPU',
                              verbose=100, 
                              early_stopping_rounds=200)

model.fit(X_train, 
          y_train, 
          eval_set=(X_valid, y_valid), 
          cat_features=['master_id','plan_id','campaign_id','imp_id','adx_app_id',
                        'adx_app_name','adx_app_pkg','adx_slot_id','agent_id',
                        'net_type','region_id','make','model','os','osv'])

# 显示各特征的 importances
importances = model.get_feature_importance()
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names)
feat_imp.sort_values(inplace=True, ascending=False)
print(feat_imp)

# 验证分数
pred_valid = model.predict_proba(X_valid)

df_metric = df_valid[['creative_image_md5', 'uid', 'click']].copy()
df_metric['pro'] = pred_valid[:, 1]
df_metric = df_metric.sort_values(['creative_image_md5', 'pro'], ascending=[True, False]).reset_index(drop=True)
display(df_metric[df_metric['click'] == 1]['pro'].describe())

df_metric = df_metric.groupby('creative_image_md5').apply(lambda g: g.head(10)).reset_index(drop=True)
df_hits = df_metric.groupby('creative_image_md5')['click'].sum().to_frame(name='hits').reset_index(drop=True)
print('recall@10 image wise:', df_hits[df_hits.hits>0].shape[0] / len(df_hits))

df_hits = df_metric.groupby('creative_image_md5')['click'].agg(list).to_frame(name='clicks').reset_index(drop=True)
scores = []
for _, row in tqdm(df_hits.iterrows(), total=len(df_hits)):
    scores.append(sum(row['clicks']) / 10)
print('recall@10 avg:', np.mean(scores))

# 全量再训练一次

X_train = pd.concat([X_train, X_valid]).reset_index(drop=True)
y_train = pd.concat([y_train, y_valid]).reset_index(drop=True)
best_iteration = model.best_iteration_ + 500    # 在原来基础上再多训练 500 轮次
del model
model = cb.CatBoostClassifier(iterations=best_iteration,
                              learning_rate=0.01,
                              depth=6,
                              eval_metric='AUC',
                              task_type='GPU',
                              verbose=100)
model.fit(X_train, 
          y_train, 
          eval_set=(X_train, y_train), 
          cat_features=['master_id','plan_id','campaign_id','imp_id','adx_app_id',
                        'adx_app_name','adx_app_pkg','adx_slot_id','agent_id',
                        'net_type','region_id','make','model','os','osv'])

# 测试集预测
test_data = test_data[test_data['creative_image_md5'].isin(test_images)].reset_index(drop=True)
test_pred = model.predict_proba(test_data[features])

# 生成提交文件
df_sub = test_data[['creative_image_md5', 'uid', 'dt']].copy()
df_sub['pro'] = test_pred[:, 1]

# 保存预测过程文件
os.makedirs('prediction', exist_ok=True)
df_sub.to_pickle('prediction/df_sub_ctb.pickle')

df_sub = df_sub.sort_values(['creative_image_md5', 'dt', 'pro'],
                            ascending=[True, True, False])

# 只需提交 top10
sub = df_sub.groupby(['creative_image_md5', 'dt']).apply(lambda g: g.head(10)).reset_index(drop=True)

# 按比赛要求的提交格式生成提交文件
os.makedirs('submit', exist_ok=True)
for d in ['2023-05-14', '2023-05-15', '2023-05-16']:
    s = sub[sub['dt'] == d].reset_index(drop=True)
    s.drop('dt', axis=1, inplace=True)
    s = s[['uid', 'creative_image_md5', 'pro']].copy()
    s.to_csv(f'submit/submission_{d}.csv')
