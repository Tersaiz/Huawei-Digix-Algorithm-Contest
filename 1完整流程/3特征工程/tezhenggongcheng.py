#!/usr/bin/python 

# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp
import matplotlib.pyplot as plt
import time
import seaborn as sns
#from untitled0 import MeanEncoder


    
    
train_df= pd.read_csv('F:/train26.csv')
test= pd.read_csv('F:/test.csv')
test['Id']= test['label']
train_df['Id'] = -1
del test['label']
data = pd.concat([train_df,test],axis=0,ignore_index=True)
#data = pd.read_csv('F:/tezheng.csv')
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                          100 * (start_mem - end_mem) / start_mem))
    return df

data = reduce_mem_usage(data)

#user_info 相关特征

#feature=[ 'slotId','adId','siteId','contentId','primId','creativeType','intertype','firstClass','spreadAppId',]
feature=[ 'city','province','phoneType','carrier','age','gender']
for f in feature:
    e = data.groupby([f])['operTime'].count().reset_index(name=f+'_count') 
    data = data.merge(e,on=f,how='left')
    data[f+'_uId_mean'] = data[f+'_count']/data['uId_count']
data['per_area'] = data['city_count']/data['province_count']  

#ad_info 相关特征
feature=[ 'adId','billId','primId','creativeType','intertype']
for f in feature:
    e = data.groupby([f])['operTime'].count().reset_index(name=f+'_count') 
    data = data.merge(e,on=f,how='left')
    data[f+'_uId_mean'] = data[f+'_count']/data['uId_count']

#content_info 相关特征
feature=['contentId','firstClass','secondClass']
for f in feature:
    e = data.groupby([f])['operTime'].count().reset_index(name=f+'_count') 
    data = data.merge(e,on=f,how='left')
    data[f+'_uId_mean'] = data[f+'_count']/data['uId_count']
data['per_class'] = data['secondClass_count']/data['firstClass_count']     
import datetime

feature=[ 'siteId','slotId','netType']
for f in feature:
    e = data.groupby([f])['operTime'].count().reset_index(name=f+'_count') 
    data = data.merge(e,on=f,how='left')
    data[f+'_uId_mean'] = data[f+'_count']/data['uId_count']
    
#del data['netType_count_x']
data['operTime']=pd.to_datetime(data['operTime'])
data['hour'] = data['operTime'].dt.hour
#data['day'] = data['operTime'].dt.day
#e = data.groupby(['uId'])['day'].nunique().reset_index(name=f+'_count')
#data = data.merge(e,on='uId',how='left')

#encoder= LabelEncoder().fit(data['day'])
#data['day'] = encoder.transform(data["day"])
#组合特征
a  = data.groupby(['uId'])['hour'].max().reset_index(name='most_time')#各用户最常访问时间段
b = data.groupby(['hour'])['uId'].count().reset_index(name='most_time_Id') #最多访问次数的时间段
data = data.merge(a,on='uId',how='left').merge(b,on='hour',how='left')

feature=[ 'billId','hour']
for f in feature:
    e = data.groupby(['uId'])[f].var().reset_index(name=f+'_var') 
    data = data.merge(e,on='uId',how='left')


#feature = ['adId','contentId']
#for f in feature:
#    e = data.groupby(['uId'])[f].nunique().reset_index(name=f+'_unique') 
#    data = data.merge(e,on='uId',how='left')

#相关性可视化
#plt.figure(figsize=(40, 32))  # 指定绘图对象宽度和高度
#colnm = data.columns.tolist()  # 列表头
#mcorr = data[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
#mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
#mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
#cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
#g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
#plt.show()
#plt.savefig('F:/2I.fig')
#  0.95041  0.962868
#加上city province 0.95031  0.962869
#按关系表删除most_time——id  per_area hour province_count city_ount   0.95021

#lgb
 # 63特征  0.97012  0.95103 0.94994 0.96763 0.96736 0.96781    0.963798
 # 50特征  0.97005  0.95101 0.94991 0.96759 0.96736 0.96781    0.963548
 #权重  0.963305
 #权重  
 #删掉mean     28  0.94756   0.962897
 # 按label相关性筛选  0.97001  0.95089  0.94992    0.96764    0.96735  0.96785  0.963586
 # 去掉uid计数 0.96951   0.95032  0.94922  0.96717  0.96693  0.96737
del data['operTime']
del data['uId']

#综合各天相关度特征系数小于10
#del data['billId_count']
#del data['intertype_count']
del data['gender_count']
#del data['netType_count']
#del data['intertype']
#del data['creativeType_count']
#del data['carrier_count']
#del data['carrier']
#del data['gender']
#del data['firstClass_count']
#del data['netType']
#del data['secondClass_count']
#于label相关性太低
#del data['most_time_Id']
#del data['hour']
#del data['netType_count']
#del data['siteId_count']
#del data['per_area']
#del data['gender_count']
#del data['age_count']
#del data['province_count']
#del data['carrier_count']
#del data['siteId']
#del data['city_count']
#del data['province']
#del data['phoneType']
#del data['slotId']
#del data['carrier']
#del data['netType']
#del data['city']

# 分布非常不一致
#del data['hour_var']
del data['secondClass_count']
del data['firstClass_count']
del data['contentId_uId_mean']   #与adid mean相关性高
#del data['contentId_count']
del data['creativeType_count']
#del data['primId_uId_mean']
del data['billId_count']
#del data['adId_uId_mean']
#del data['adId_count']
del data['per_area']
#del data['spreadAppId']
del data['secondClass']
del data['intertype']
#del data['firstClass']
#del data['creativeType']
del data['contentId']   #与contentID count重复率高
#del data['billId']
#del data['adId']
del data['uId_count']
del data['netType_count'] #与nettype重复率高
del data['contentId_count']  #与adid重复率高，分布差
del data['secondClass_uId_mean']  #与firstclassuidmean 重复率高
 #lgb


data = data.fillna(-1)
train_df = data[data['Id']<0]
test = data[data['Id']>0]

train_df.to_csv('train.csv',index = False)
test.to_csv('test.csv')
