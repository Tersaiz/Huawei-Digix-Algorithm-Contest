import xgboost as xgb
from sklearn import metrics, preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
import sys

input_train = sys.argv[1]
input_test = sys.argv[2]
output_name = sys.argv[3]

INPUT_TRAIN = 'by_data_50w/' + input_train +'.csv'
INPUT_TEST = 'by_data_50w/' + input_test +'.csv'



data= pd.read_csv(INPUT_TRAIN,iterator=True)
train_df = data.get_chunk(10000)
test_df= pd.read_csv(INPUT_TEST)


date = pd.to_datetime(train_df['operTime'])
x=date.dt.hour
train_df['operTime'] = x

date = pd.to_datetime(test_df['operTime'])
x=date.dt.hour
test_df['operTime'] = x


# 计数特征
len_train = train_df.shape[0]
data = pd.concat([train_df,test_df])

feature=[ 'slotId','phoneType','adId','city','operTime']
for f in feature:
    count1 = data.groupby([f])['uId'].count().reset_index(name=f+'_count') 
    data = data.merge(count1,on=f,how='left')


data.drop(['uId'],axis = 1,inplace = True)

train_df = data.iloc[0:len_train,:]
test_df = data.iloc[len_train:,:]

features = [c for c in train_df.columns if c not in ['label']]
target = train_df['label']



params = {'objective': 'binary:logistic',
               'eval_metric': 'auc',
               'max_depth': 14,
               'eta': 0.1,
               'gamma': 6,
               'subsample': 0.9,
               'colsample_bytree': 0.9,
               'min_child_weight': 51,
               'colsample_bylevel': 0.6,
               'lambda': 0.5,
               'alpha': 0.1,
               'silent':0}
    
    
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=2019)
oof =  np.zeros(len(train_df)) 
predictions =np.zeros(len(test_df))

for i, (trn, val) in enumerate(folds.split(train_df.values,target.values)):
    print(i+1, "fold.    AUC")
    
    trn_x = train_df.iloc[trn][features]
    trn_y = target.iloc[trn]
    val_x = train_df.iloc[val][features]
    val_y = target.iloc[val]

    

    model = xgb.train(params
                      , xgb.DMatrix(trn_x, trn_y)
                      , 100000
                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]
                      , verbose_eval=5000
                      , early_stopping_rounds=3000
                      )

    oof[val] = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)
    predictions += model.predict(xgb.DMatrix(test_df[features]), ntree_limit=model.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cv_auc = roc_auc_score(target, oof)
cv_auc = cv_auc.round(6)


OUTPUT_FILE = 'xgb_result_50w/' + output_name +'-'+ str(cv_auc) + '.csv'
sub_df = pd.DataFrame({"id":test_df["label"].values})
sub_df["probability"] = predictions.round(6)
sub_df.to_csv(OUTPUT_FILE, index=False)

