import pandas as pd
import numpy as np
import tensorflow as tf
import ctrNet
from sklearn.model_selection import train_test_split
from src import misc_utils as utils
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys

input_data = sys.argv[1]
output_data = sys.argv[2]

input_path = 'data/' + input_data +'.csv'
output_path = 'result_best/XdeepFM_' + output_data + '_'


train_df=pd.read_csv(input_path)
test_df=pd.read_csv('data/uid1_label1_test_data.csv')

date = pd.to_datetime(train_df['operTime'])
x=date.dt.hour
train_df['operTime'] = x

date = pd.to_datetime(test_df['operTime'])
x=date.dt.hour
test_df['operTime'] = x



len_train = train_df.shape[0]
data = pd.concat([train_df,test_df])

feature=[ 'slotId','phoneType','adId','city','operTime']
for f in feature:
    count1 = data.groupby([f])['uId'].count().reset_index(name=f+'_count') 
    data = data.merge(count1,on=f,how='left')


data.drop(['uId'],axis = 1,inplace = True)

train_df = data.iloc[0:len_train,:]
test_df = data.iloc[len_train:,:]



sub = pd.DataFrame()
sub['id'] = test_df['label']
test_df['label']  = -1


features=[i for i in train_df.columns if i not in ['label']]


folds = StratifiedKFold(n_splits=3, shuffle=False, random_state=44000)

# oof_FM = np.zeros(len(train_df))
# preds_FM = np.zeros(len(test_df))

# oof_FFM = np.zeros(len(train_df))
# preds_FFM = np.zeros(len(test_df))

# oof_FNFM = np.zeros(len(train_df))
# preds_NFFM = np.zeros(len(test_df))

oof_XDEEPFM = np.zeros(len(train_df))
preds_XDEEPFM = np.zeros(len(test_df))

# train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.1, random_state=2019)


# #FM
# hparam=tf.contrib.training.HParams(
#             model='fm', #['fm','ffm','nffm']
#             k=16,
#             hash_ids=int(1e5),
#             batch_size=1024,
#             optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
#             learning_rate=0.0002,
#             num_display_steps=1000,
#             num_eval_steps=1000,
#             # steps=200,
#             epoch=10,
#             metric='auc', #['auc','logloss']
#             init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
#             init_value=0.1,
#             feature_nums=len(features))
# utils.print_hparams(hparam)
# os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# model=ctrNet.build_model(hparam)
# print("Testing FM....")
# model.train(train_data=(train_df[features],train_df['label']),\
#             dev_data=(dev_df[features],dev_df['label']))
# # from sklearn import metrics
# preds=model.infer(dev_data=(test_df[features],test_df['label']))
# # fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
# # auc=metrics.auc(fpr, tpr)
# # print(preds)
# preds = np.round(preds,6)
# sub['probability'] = preds
# sub.to_csv('result_28/submission_'+'FM'+'.csv', index=False)



# print("FM Done....")




# #FFM
# hparam=tf.contrib.training.HParams(
#             model='ffm', #['fm','ffm','nffm']
#             k=16,
#             hash_ids=int(1e5),
#             batch_size=1024,
#             optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
#             learning_rate=0.0002,
#             num_display_steps=1000,
#             num_eval_steps=1000,
#             epoch=10,
#             metric='auc', #['auc','logloss']
#             init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
#             init_value=0.1,
#             feature_nums=len(features))
# utils.print_hparams(hparam)
# os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# model=ctrNet.build_model(hparam)
# print("Testing FFM....")
# model.train(train_data=(train_df[features],train_df['label']),\
#             dev_data=(dev_df[features],dev_df['label']))
# # from sklearn import metrics
# preds=model.infer(dev_data=(test_df[features],test_df['label']))
# # fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
# # auc=metrics.auc(fpr, tpr)
# # print(auc)
# preds = np.round(preds,6)
# sub['probability'] = preds
# sub.to_csv('result_28/submission_'+'FFM'+'.csv', index=False)


# print("FFM Done....")

# #NFFM
# hparam=tf.contrib.training.HParams(
#             model='nffm',
#             norm=True,
#             batch_norm_decay=0.9,
#             hidden_size=[128,128],
#             cross_layer_sizes=[128,128,128],
#             k=8,
#             hash_ids=int(2e5),
#             batch_size=1024,
#             optimizer="adam",
#             learning_rate=0.001,
#             num_display_steps=1000,
#             num_eval_steps=1000,
#             epoch=10,
#             metric='auc',
#             activation=['relu','relu','relu'],
#             cross_activation='identity',
#             init_method='uniform',
#             init_value=0.1,
#             feature_nums=len(features))
# utils.print_hparams(hparam)
# os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# model=ctrNet.build_model(hparam)

# print("Testing NFFM....")

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['label'].values)):

#     print("Fold {}".format(fold_))

#     train = train_df.iloc[trn_idx]
#     dev = train_df.iloc[val_idx]


#     model.train(train_data=(train[features],train['label']),dev_data=(dev[features],dev['label']))

#     oof_FNFM[val_idx] = model.infer(dev_data=(dev[features],dev['label']))
#     preds_NFFM += model.infer(dev_data=(test_df[features],test_df['label']))/ folds.n_splits

# print("CV score: {:<8.5f}".format(roc_auc_score(train_df['label'], oof_FNFM)))
# cv_auc = roc_auc_score(train_df['label'], oof_FNFM)
# cv_auc = cv_auc.round(6)

# preds_NFFM = np.round(preds_NFFM,6)
# sub['probability'] = preds_NFFM
# sub.to_csv('result_best/NFFM'+str(cv_auc) + '.csv', index=False)


# print("NFFM Done....")




# #Xdeepfm
hparam=tf.contrib.training.HParams(
            model='xdeepfm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            cross_layer_sizes=[128,128,128],
            k=8,
            hash_ids=int(2e5),
            batch_size=1024,
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=10,
            metric='auc',
            activation=['relu','relu','relu'],
            cross_activation='identity',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features))
utils.print_hparams(hparam)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
model=ctrNet.build_model(hparam)
print("Testing XdeepFM....")

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['label'].values)):

    print("Fold {}".format(fold_))

    train = train_df.iloc[trn_idx]
    dev = train_df.iloc[val_idx]


    model.train(train_data=(train[features],train['label']),dev_data=(dev[features],dev['label']))



    oof_XDEEPFM[val_idx] = model.infer(dev_data=(dev[features],dev['label']))
    # from sklearn import metrics
    preds_XDEEPFM += model.infer(dev_data=(test_df[features],test_df['label']))/ folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(train_df['label'], oof_XDEEPFM)))
cv_auc = roc_auc_score(train_df['label'], oof_XDEEPFM)
cv_auc = cv_auc.round(6)

preds_XDEEPFM = np.round(preds_XDEEPFM,6)
sub['probability'] = preds_XDEEPFM
sub.to_csv(output_path+str(cv_auc) + '.csv', index=False)

print("XdeepFM Done....")

