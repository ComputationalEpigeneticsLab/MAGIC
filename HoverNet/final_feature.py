import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
import pickle



def process_single_patch(patch_df, all_columns,all_features):
    """
     Arguments:
        patch_df: A DataFrame containing the nuclear features for a single patch
        all_columns: All possible column names
        all_features: All features
    Returns:
        pd.Series: A feature vector containing the calculated statistics
    """
    #
    stats = {col: 0 for col in all_columns}

    if not patch_df.empty:
        #
        grouped = patch_df.groupby('type')

        #
        for cell_type, group in grouped:

            stats[f'type_{cell_type}_count'] = len(group)

            for col in all_features:
                stats[f'type_{cell_type}_{col}'] = group[col].mean()

    #
    return pd.Series(stats)[all_columns]

#
removed_list_path = '/hovernet/removed_feature_list.txt'
with open(removed_list_path, 'r') as file:
    removed_feature_list = [line.strip() for line in file]

samples_list = glob.glob("/hovernet/nuclei_standar_features/train/*/*/*")
final_root_path= '/train_test_split'

file_path_1 = glob.glob(os.path.join(samples_list[0],"*"))[0]
sample_1 = pd.read_csv(file_path_1,index_col=0)
all_features = sample_1.drop(['Identifier.CentroidX', 'Identifier.CentroidY','type'] + removed_feature_list, axis=1).columns
#
columns = []
#1:tumor cells; 2:lymphocytes; 3:connective tissue(Stromal cells)
all_cell_types = ['1','2','3']
for cell_type in sorted(all_cell_types):
        #
        columns.append(f'type_{cell_type}_count')
        #
        for feature in all_features:
            columns.append(f'type_{cell_type}_{feature}')


all_sample_stats = []
for sample in samples_list:

    sample_save_path = os.path.join(final_root_path,'hovernet/train',sample.split("/")[-3],sample.split("/")[-2],sample.split("/")[-1])

    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    patches_name = []
    sample_stats = []
    for feature_file in os.listdir(sample):
        patch_name = feature_file.split('.')[0]
        patches_name.append(patch_name)
        feature_path = os.path.join(sample,feature_file)
        feature_df = pd.read_csv(feature_path,index_col=0)
        patch_stats = process_single_patch(feature_df, columns,all_features)
        sample_stats.append(patch_stats)
        all_sample_stats.append(patch_stats)


    wsi_matrix = pd.DataFrame(sample_stats)
    wsi_matrix.index = patches_name

    wsi_matrix.to_csv(os.path.join(sample_save_path,'patch_features.csv'))
all_sample_stats = pd.DataFrame(all_sample_stats)
count_name = ['type_1_count','type_2_count','type_3_count']
all_sample_stats[count_name] = np.log1p(all_sample_stats[count_name])
scaler = StandardScaler().fit(all_sample_stats[count_name])
with open('/hovernet/scale_cell_type_count.pkl','wb') as f:
    pickle.dump(scaler, f)

#
samples_list = glob.glob(final_root_path+'/hovernet/train/*/*/*/*')
for sample in samples_list:
    feature_df = pd.read_csv(sample, index_col=0)
    feature_df[count_name] = np.log1p(feature_df[count_name])
    feature_df[count_name] = scaler.transform(feature_df[count_name])
    feature_df.to_csv(sample)
del all_sample_stats

########internal_test
samples_list = glob.glob("/hovernet/nuclei_standar_features/internal_test/*/*/*")
final_root_path= '/train_test_split'
for sample in samples_list:

    sample_save_path = os.path.join(final_root_path,'hovernet/internal_test',sample.split("/")[-3],sample.split("/")[-2],sample.split("/")[-1])

    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    patches_name = []
    sample_stats = []
    for feature_file in os.listdir(sample):
        patch_name = feature_file.split('.')[0]
        patches_name.append(patch_name)
        feature_path = os.path.join(sample,feature_file)
        feature_df = pd.read_csv(feature_path,index_col=0)
        patch_stats = process_single_patch(feature_df, columns,all_features)
        sample_stats.append(patch_stats)


    wsi_matrix = pd.DataFrame(sample_stats)
    wsi_matrix.index = patches_name
    wsi_matrix[count_name] = np.log1p(wsi_matrix[count_name])
    wsi_matrix[count_name] = scaler.transform(wsi_matrix[count_name])

    wsi_matrix.to_csv(os.path.join(sample_save_path,'patch_features.csv'))



########fold-5 cv
for ii in list(np.arange(5)):
    samples_list = glob.glob(f'/hovernet/nuclei_standar_features/fold_{ii}/train/*/*/*')
    final_root_path= f'/train_test_split/5_cv_split/fold_{ii}'
    all_sample_stats = []
    for sample in samples_list:

        sample_save_path = os.path.join(final_root_path,'hovernet/train',sample.split("/")[-3],sample.split("/")[-2],sample.split("/")[-1])

        if not os.path.exists(sample_save_path):
            os.makedirs(sample_save_path)

        patches_name = []
        sample_stats = []
        for feature_file in os.listdir(sample):
            patch_name = feature_file.split('.')[0]
            patches_name.append(patch_name)
            feature_path = os.path.join(sample,feature_file)
            feature_df = pd.read_csv(feature_path,index_col=0)
            patch_stats = process_single_patch(feature_df, columns,all_features)
            sample_stats.append(patch_stats)
            all_sample_stats.append(patch_stats)

        wsi_matrix = pd.DataFrame(sample_stats)
        wsi_matrix.index = patches_name

        wsi_matrix.to_csv(os.path.join(sample_save_path,'patch_features.csv'))

    all_sample_stats = pd.DataFrame(all_sample_stats)
    all_sample_stats[count_name] = np.log1p(all_sample_stats[count_name])
    scaler = StandardScaler().fit(all_sample_stats[count_name])
    with open(f'/hovernet/nuclei_standar_features/fold_{ii}/scale_cell_type_count.pkl', 'wb') as f:
        pickle.dump(scaler, f)


    #
    samples_list = glob.glob(final_root_path+'/hovernet/train/*/*/*/*')
    for sample in samples_list:
        feature_df = pd.read_csv(sample, index_col=0)
        feature_df[count_name] = np.log1p(feature_df[count_name])
        feature_df[count_name] = scaler.transform(feature_df[count_name])
        feature_df.to_csv(sample)

    del all_sample_stats

    #val data
    samples_list = glob.glob(f'/hovernet/nuclei_standar_features/fold_{ii}/val/*/*/*')
    final_root_path = f'/train_test_split/5_cv_split/fold_{ii}'
    for sample in samples_list:

        sample_save_path = os.path.join(final_root_path, 'hovernet/val', sample.split("/")[-3],sample.split("/")[-2], sample.split("/")[-1])

        if not os.path.exists(sample_save_path):
            os.makedirs(sample_save_path)

        patches_name = []
        sample_stats = []
        for feature_file in os.listdir(sample):
            patch_name = feature_file.split('.')[0]
            patches_name.append(patch_name)
            feature_path = os.path.join(sample, feature_file)
            feature_df = pd.read_csv(feature_path, index_col=0)
            patch_stats = process_single_patch(feature_df, columns, all_features)
            sample_stats.append(patch_stats)

        wsi_matrix = pd.DataFrame(sample_stats)
        wsi_matrix.index = patches_name
        wsi_matrix[count_name] = np.log1p(wsi_matrix[count_name])
        wsi_matrix[count_name] = scaler.transform(wsi_matrix[count_name])
        wsi_matrix.to_csv(os.path.join(sample_save_path, 'patch_features.csv'))



