# 讯飞AI开发者大赛 移动广告营销场景下的人群召回算法挑战赛 top1 代码

## 方案概述

1. 对数据进行分析并构建特征及树模型进行建模
2. 特征工程：groupby 统计特征、类别模型 nunique、时间特征等；分为 item 侧及 user 侧分别构建
3. 验证划分：训练集最后一天作为验证集，对训练样本进行负采样；并最后采用在线全量训练
4. 类型转换减少内存占用

## 硬件要求

1. 内存 128G 及以上
2. 显存 12G 以上
3. 磁盘空间 100G
4. CPU 无特殊要求

## 目录结构

```
.
├── 01_data_prepare.py
├── 02_make_features.py
├── 03_run_catboost.py
├── README.md
├── xfdata
│   ├── submit
│   ├── test_dataset
│   └── train_dataset
└── run.sh
```

## 复现步骤

1. 将比赛数据按要求放到 data 目录下 (train_dataset: 训练集; test_dataset: 测试集; submit: 提交示例)
2. 执行 bash run.sh 即可
