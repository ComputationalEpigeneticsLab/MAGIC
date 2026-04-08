# Standardize numerical features of nucleus at the WSI level
import os
import math
import argparse
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import glob

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle





feature_root_path = '/hovernet/nuclei_features'
norm_root_path = '/hovernet/nuclei_standar_features'
if not os.path.exists(norm_root_path):
    os.makedirs(norm_root_path)

#
train_sample =pd.read_csv(os.path.join('/train_test_split','train_response.csv'))
train_sample.columns = ["filepath"]
train_sample1 = train_sample['filepath'].to_list()
#
train_sample =pd.read_csv(os.path.join('/train_test_split','train_resistant.csv'))
train_sample.columns = ["filepath"]
train_sample1.extend(train_sample['filepath'].to_list())
#
train_samples = [os.path.basename(path) for path in train_sample1]#Example: /lunit_result/response（Classification）/2351301（Patient ID)

samples_list = glob.glob("/hovernet/nuclei_features/*/*/*/*")#Example: hovernet/nuclei_features/response（Classification）/2351301（Patient ID)/1（WSI ID）/top_left_coords_53958_126336_.png
positions = [i for item in train_samples
             for i, x in enumerate(samples_list)
             if item in x]
samples_list_train_1 = np.array(samples_list)
def is_valid_csv(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

samples_list_train_1 = samples_list_train_1[positions].tolist()
samples_list_train = [f for f in samples_list_train_1 if is_valid_csv(f)]
samples_list_train.sort()
samples1 = [x.split('/')[-3] for x in samples_list_train]
samples1 = set(samples1)
len(samples1) ==len(train_samples)

del samples_list_train_1
#
from multiprocessing import Pool
import pandas as pd

def read_csv_and_add_barcode(feature_path):
    patch_name = feature_path.split('.')[0]
    try:
        df = pd.read_csv(feature_path, index_col=0)
        if df.empty:
            print(f"Skipping empty file: {feature_path}")
            return None
        df.insert(0, 'barcode', patch_name)
        return df
    except Exception as e:
        print(f"Error reading {feature_path}: {e}")
        return None

#
with Pool(processes=220) as pool:
    dfs = pool.map(read_csv_and_add_barcode, samples_list_train)
    dfs = [df for df in dfs if df is not None]


tissue_feature_df = pd.concat(dfs, axis=0)
del dfs
tissue_feature_df['type'] = tissue_feature_df['type'].apply(str)
#Consider only the tumor, stroma, and inflammatory cells
tissue_feature_df = tissue_feature_df[tissue_feature_df['type'].isin(['1', '2', '3'])]
scale_features = tissue_feature_df.columns[4:]
hovernet_features = pd.DataFrame(scale_features,columns=["hovernet_features"])
hovernet_features.to_csv("/hovernet_features.csv")
#NA Value Filling
na_counts = tissue_feature_df.isna().sum()
type_means = tissue_feature_df.groupby('type')[scale_features].mean()
# saving
type_means.to_csv("/hovernet/all_train_cell_type_means.csv")
#Filling
for col in scale_features:
    tissue_feature_df[col] = tissue_feature_df.groupby('type')[col].transform(
        lambda x: x.fillna(x.mean())
    )
#
scaler = StandardScaler().fit(tissue_feature_df[scale_features])
with open('/hovernet/nuclei_standar_features/train/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

tissue_feature_df[scale_features] = scaler.transform(tissue_feature_df[scale_features])


#Calculate the correlations between features and select features (using all training samples)
#
nuclei_corr_matrix = tissue_feature_df[scale_features].corr(method='spearman')
nuclei_corr_matrix.to_csv('/hovernet/features_cor.csv')

threshold = 0.9
removed_features = []
for feature1 in nuclei_corr_matrix.columns:
    for feature2 in nuclei_corr_matrix.columns:
        if feature1 != feature2 and abs(nuclei_corr_matrix.loc[feature1, feature2]) > threshold:
            mean_abs_corr_feature1 = abs(nuclei_corr_matrix[feature1]).mean()
            mean_abs_corr_feature2 = abs(nuclei_corr_matrix[feature2]).mean()
            if mean_abs_corr_feature1 > mean_abs_corr_feature2:
                removed_features.append(feature1)
            else:
                removed_features.append(feature2)

removed_feature_list = list(set(removed_features))
print("len(removed_feature_list):", len(removed_feature_list))
with open("/hovernet/removed_feature_list.txt", "w") as file:
    for item in removed_feature_list:
        file.write(item + "\n")

#Store by patch
barcode_groups = tissue_feature_df.groupby(tissue_feature_df.barcode)
for feature_file in samples_list_train:
    patch_name = feature_file.split('.')[0]
    if patch_name not in barcode_groups.groups:
        #print(f"Warning: Patch {patch_name} not found in barcode groups, skipping...")
        continue
    norm_feature_path = os.path.join(norm_root_path, 'train',feature_file.split("/")[-4], feature_file.split("/")[-3],
                                        feature_file.split("/")[-2])
    if not os.path.exists(norm_feature_path):
        os.makedirs(norm_feature_path)
    normalized_path = os.path.join(norm_feature_path, feature_file.split('/')[-1])
    normalized_df = barcode_groups.get_group(patch_name).copy()
    normalized_df.drop(columns='barcode', inplace=True)
    if normalized_df.isnull().any().any():
        print("normalized_df.isnull().any().any()--path:", normalized_path)

    normalized_df.to_csv(normalized_path, header=True, float_format='%.3f')

del tissue_feature_df,samples_list_train,barcode_groups



#test data
train_sample =pd.read_csv(os.path.join('/train_test_split','internal_test_response.csv'))
train_sample.columns = ["filepath"]
train_sample1 = train_sample['filepath'].to_list()
#
train_sample =pd.read_csv(os.path.join('/train_test_split','internal_test_resistant.csv'))
train_sample.columns = ["filepath"]
train_sample1.extend(train_sample['filepath'].to_list())
#
train_samples = [os.path.basename(path) for path in train_sample1] ##Example: /lunit_result/response（Classification）/2351301（Patient ID)

positions = [i for item in train_samples
             for i, x in enumerate(samples_list)
             if item in x]
samples_list_test_1 = np.array(samples_list)
samples_list_test_1 = samples_list_test_1[positions].tolist()
samples_list_test = [f for f in samples_list_test_1 if is_valid_csv(f)]
samples_list_test.sort()
samples1 = [x.split('/')[-3] for x in samples_list_test]
samples1 = set(samples1)
len(samples1) ==len(train_samples)
del samples_list_test_1
#
with Pool(processes=220) as pool:
    dfs = pool.map(read_csv_and_add_barcode, samples_list_test)
    dfs = [df for df in dfs if df is not None]

tissue_feature_test = pd.concat(dfs, axis=0)
del dfs

tissue_feature_test['type'] = tissue_feature_test['type'].apply(str)
#
tissue_feature_test = tissue_feature_test[tissue_feature_test['type'].isin(['1', '2', '3'])]
#
nuclei_test_filled = tissue_feature_test.copy()
for cell_type in type_means.index:
    mask = (tissue_feature_test['type'] == cell_type)
    nuclei_test_filled.loc[mask] = nuclei_test_filled.loc[mask].fillna(type_means.loc[cell_type])
#
tissue_feature_test = nuclei_test_filled
del nuclei_test_filled
tissue_feature_test[scale_features] = scaler.transform(tissue_feature_test[scale_features])

#
barcode_groups = tissue_feature_test.groupby(tissue_feature_test.barcode)
for feature_file in samples_list_test:
    patch_name = feature_file.split('.')[0]
    if patch_name not in barcode_groups.groups:
        #print(f"Warning: Patch {patch_name} not found in barcode groups, skipping...")
        continue
    norm_feature_path = os.path.join(norm_root_path, 'internal_test',feature_file.split("/")[-4], feature_file.split("/")[-3],
                                        feature_file.split("/")[-2])
    if not os.path.exists(norm_feature_path):
        os.makedirs(norm_feature_path)
    normalized_path = os.path.join(norm_feature_path, feature_file.split('/')[-1])
    normalized_df = barcode_groups.get_group(patch_name).copy()
    normalized_df.drop(columns='barcode', inplace=True)
    if normalized_df.isnull().any().any():
        print("normalized_df.isnull().any().any()--path:", normalized_path)

    normalized_df.to_csv(normalized_path, header=True, float_format='%.3f')

del samples_list_test,tissue_feature_test,barcode_groups


############5-fold cv
for ii in list(np.arange(5)):

    train_sample =pd.read_csv(os.path.join('/train_test_split/5_cv_split',f'fold_{ii}','train_data.csv'))
    train_sample.columns = ["filepath"]
    train_sample1 = train_sample['filepath'].to_list()
    #
    train_samples = [os.path.basename(path) for path in train_sample1]

    positions = [i for item in train_samples
             for i, x in enumerate(samples_list)
             if item in x]
    samples_list_train_1 = np.array(samples_list)
    samples_list_train_1 = samples_list_train_1[positions].tolist()
    samples_list_train = [f for f in samples_list_train_1 if is_valid_csv(f)]
    samples_list_train.sort()
    del samples_list_train_1
    #
    with Pool(processes=220) as pool:
        dfs = pool.map(read_csv_and_add_barcode, samples_list_train)
        dfs = [df for df in dfs if df is not None]

    tissue_feature_df = pd.concat(dfs, axis=0)
    del dfs

    tissue_feature_df['type'] = tissue_feature_df['type'].apply(str)
    #
    tissue_feature_df = tissue_feature_df[tissue_feature_df['type'].isin(['1', '2', '3'])]
    #
    type_means = tissue_feature_df.groupby('type')[scale_features].mean()
    # saving
    type_means.to_csv(f'/hovernet/fold_{ii}_cell_type_means.csv')
    #
    for col in scale_features:
        tissue_feature_df[col] = tissue_feature_df.groupby('type')[col].transform(
            lambda x: x.fillna(x.mean())
        )
    #
    scaler = StandardScaler().fit(tissue_feature_df[scale_features])
    with open(f'/hovernet/nuclei_standar_features/fold_{ii}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    tissue_feature_df[scale_features] = scaler.transform(tissue_feature_df[scale_features])

    ###################
    barcode_groups = tissue_feature_df.groupby(tissue_feature_df.barcode)
    for feature_file in samples_list_train:
        patch_name = feature_file.split('.')[0]
        if patch_name not in barcode_groups.groups:
            # print(f"Warning: Patch {patch_name} not found in barcode groups, skipping...")
            continue
        norm_feature_path = os.path.join(norm_root_path, f'fold_{ii}/train',feature_file.split("/")[-4], feature_file.split("/")[-3],
                                        feature_file.split("/")[-2])
        if not os.path.exists(norm_feature_path):
            os.makedirs(norm_feature_path)
        normalized_path = os.path.join(norm_feature_path, feature_file.split('/')[-1])
        normalized_df = barcode_groups.get_group(patch_name).copy()
        normalized_df.drop(columns='barcode', inplace=True)
        if normalized_df.isnull().any().any():
            print("normalized_df.isnull().any().any()--path:", normalized_path)

        normalized_df.to_csv(normalized_path, header=True, float_format='%.3f')

    del tissue_feature_df,samples_list_train,barcode_groups

    #test data
    train_sample =pd.read_csv(os.path.join('/train_test_split/5_cv_split',f'fold_{ii}','val_data.csv'))
    train_sample.columns = ["filepath"]
    train_sample1 = train_sample['filepath'].to_list()
    #
    train_samples = [os.path.basename(path) for path in train_sample1]

    positions = [i for item in train_samples
                for i, x in enumerate(samples_list)
                if item in x]
    samples_list_test_1 = np.array(samples_list)
    samples_list_test_1 = samples_list_test_1[positions].tolist()
    samples_list_test = [f for f in samples_list_test_1 if is_valid_csv(f)]
    samples_list_test.sort()
    del samples_list_test_1
    #
    with Pool(processes=210) as pool:
        dfs = pool.map(read_csv_and_add_barcode, samples_list_test)
        dfs = [df for df in dfs if df is not None]

    tissue_feature_test = pd.concat(dfs, axis=0)
    del dfs


    tissue_feature_test['type'] = tissue_feature_test['type'].apply(str)
    #
    tissue_feature_test = tissue_feature_test[tissue_feature_test['type'].isin(['1', '2', '3'])]
    #
    nuclei_test_filled = tissue_feature_test.copy()
    for cell_type in type_means.index:
        mask = (tissue_feature_test['type'] == cell_type)
        nuclei_test_filled.loc[mask] = nuclei_test_filled.loc[mask].fillna(type_means.loc[cell_type])
    #
    tissue_feature_test = nuclei_test_filled
    del nuclei_test_filled
    tissue_feature_test[scale_features] = scaler.transform(tissue_feature_test[scale_features])

    #
    barcode_groups = tissue_feature_test.groupby(tissue_feature_test.barcode)
    for feature_file in samples_list_test:
        patch_name = feature_file.split('.')[0]
        if patch_name not in barcode_groups.groups:
            # print(f"Warning: Patch {patch_name} not found in barcode groups, skipping...")
            continue
        norm_feature_path = os.path.join(norm_root_path, f'fold_{ii}/val',feature_file.split("/")[-4], feature_file.split("/")[-3],
                                            feature_file.split("/")[-2])
        if not os.path.exists(norm_feature_path):
            os.makedirs(norm_feature_path)
        normalized_path = os.path.join(norm_feature_path, feature_file.split('/')[-1])
        normalized_df = barcode_groups.get_group(patch_name).copy()
        normalized_df.drop(columns='barcode', inplace=True)
        if normalized_df.isnull().any().any():
            print("normalized_df.isnull().any().any()--path:", normalized_path)

        normalized_df.to_csv(normalized_path, header=True, float_format='%.3f')

    del samples_list_test,tissue_feature_test