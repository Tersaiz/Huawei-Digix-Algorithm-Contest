#!/usr/bin/python 

# -*- coding: utf-8 -*-

import pandas as pd
import time
import gc
import datetime
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp
import matplotlib.pyplot as plt
import time
import sys
import seaborn as sns
encoder=preprocessing.LabelEncoder()




input_train = sys.argv[1]
input_test = sys.argv[2]
output_name = sys.argv[3]

INPUT_TRAIN = 'by_data_50w/' + input_train +'.csv'
INPUT_TEST = 'by_data_50w/' + input_test +'.csv'
OUTPUT_PIC = 'result_50w/' + output_name +'.png'


train_df= pd.read_csv(INPUT_TRAIN)
test= pd.read_csv(INPUT_TEST)


date = pd.to_datetime(train_df['operTime'])
x=date.dt.hour
train_df['operTime'] = x

date = pd.to_datetime(test['operTime'])
x=date.dt.hour
test['operTime'] = x


# 计数特征
len_train = train_df.shape[0]
data = pd.concat([train_df,test])

feature=[ 'slotId','phoneType','adId','city','operTime']
for f in feature:
    count1 = data.groupby([f])['uId'].count().reset_index(name=f+'_count') 
    data = data.merge(count1,on=f,how='left')


data.drop(['uId'],axis = 1,inplace = True)

train_df = data.iloc[0:len_train,:]
test = data.iloc[len_train:,:]

#lgb
features = [c for c in train_df.columns if c not in ['label']]
target = train_df['label']
param = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 50,
    'max_depth':-1,
    'min_child_samples': 121,
    'max_bin': 15,
#     'subsample': .7,
#     'subsample_freq': 1,
#     'colsample_bytree': 0.7,
#     'min_child_weight': 0,
#     'scale_pos_weight': 0.43,
    'seed': 2019,
    'nthread': 6,
    'verbose': 0,
        }
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cv_auc = roc_auc_score(target, oof)
cv_auc = cv_auc.round(6)


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig(OUTPUT_PIC)



OUTPUT_FILE = 'result_50w/' + output_name +'-'+ str(cv_auc) + '.csv'
sub_df = pd.DataFrame({"id":test["label"].values})
sub_df["probability"] = predictions.round(6)
sub_df.to_csv(OUTPUT_FILE, index=False)