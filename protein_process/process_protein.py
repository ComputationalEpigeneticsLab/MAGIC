import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


relative_exp = pd.read_csv('/report.pg_matrix.csv',index_col=0)#protein_exp from DIA-NN
#Remove proteins that are missing in 30% of the samples
na_ratio = relative_exp.isna().mean(axis=1)
relative_exp_filtered = relative_exp[na_ratio <= 0.15]

#Missing Value Imputation
q = 0.01
all_sample = relative_exp_filtered.columns.tolist()
for i in all_sample:
    min_value = relative_exp_filtered.loc[:,i].quantile(q)
    relative_exp_filtered.loc[:,i] = relative_exp_filtered.loc[:,i].fillna(min_value)

relative_exp_filtered.to_csv(os.path.join(save_path,'relative_exp_filtered.csv'))

####
#globals().clear()
#relative_exp_filtered = pd.read_csv("/relative_exp_filtered.csv",index_col=0)
relative_exp_filtered = relative_exp_filtered.T

train_sample =pd.read_csv(os.path.join('/train_test_split','train_response.csv'))
train_sample.columns = ["filepath"]
train_sample1 = train_sample['filepath'].to_list()
#
train_sample =pd.read_csv(os.path.join('/train_test_split','train_resistant.csv'))
train_sample.columns = ["filepath"]
train_sample1.extend(train_sample['filepath'].to_list())
#
train_samples = [os.path.basename(path) for path in train_sample1]
train_data = relative_exp_filtered.loc[train_samples,]


######
val_sample =pd.read_csv(os.path.join('/train_test_split','internal_test_response.csv'))
val_sample.columns = ["filepath"]
val_sample1 = val_sample['filepath'].to_list()
#
val_sample =pd.read_csv(os.path.join('/train_test_split','internal_test_resistant.csv'))
val_sample.columns = ["filepath"]
val_sample1.extend(val_sample['filepath'].to_list())
#
val_samples = [os.path.basename(path) for path in val_sample1]
internal_test_data = relative_exp_filtered.loc[val_samples,]

#
protein = pd.read_csv("/xgboost/selected_proteins.csv")
protein = protein.loc[:,"selected_proteins"].to_list()
train_data = train_data.loc[:,protein]

internal_test_data = internal_test_data.loc[:,protein]
scaler = StandardScaler().fit(train_data)
train_data_scaled = scaler.transform(train_data)
val_data_scaled = scaler.transform(internal_test_data)
train_data = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)
val_data = pd.DataFrame(val_data_scaled, index=internal_test_data.index, columns=internal_test_data.columns)
train_data = train_data.T
val_data = val_data.T

train_data.to_csv("/train_test_split/train_data.csv")
val_data.to_csv("/train_test_split/internal_test_data.csv")

import pickle
with open('/train_test_split/protein_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#######
#globals().clear()
cv_split_path='/train_test_split/5_cv_split'
#z-score
for i in range(5):
    train_sample =pd.read_csv(os.path.join(cv_split_path,f'fold_{i}','train_data.csv'))
    train_sample = train_sample['filepath'].to_list()
    train_samples = [os.path.basename(path) for path in train_sample]
    train_data = relative_exp_filtered.loc[train_samples,protein]

    val_sample = pd.read_csv(os.path.join(cv_split_path, f'fold_{i}', 'val_data.csv'))
    val_sample = val_sample ['filepath'].to_list()
    val_samples = [os.path.basename(path) for path in val_sample]
    val_data = relative_exp_filtered.loc[val_samples,protein]

    scaler = StandardScaler().fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    train_data = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)
    val_data = pd.DataFrame(val_data_scaled, index=val_data.index, columns=val_data.columns)
    train_data = train_data.T
    val_data = val_data.T
    train_data.to_csv(os.path.join(cv_split_path,f'fold_{i}','train_protein.csv'))
    val_data.to_csv(os.path.join(cv_split_path, f'fold_{i}', 'val_protein.csv'))


