# Extract a set of numerical features for each nuclei using HistomicsTK package.
import os
import sys
import scipy.io as sio
import skimage.io
import numpy as np
import pandas as pd
import glob
import json



sys.path.append("/HistomicsTK")
import histomicstk as htk


#https://github.com/ruitian-olivia/IGI-DL/blob/master/preprocessing/nuclei_features_extract.py
def extract_seg_features(img_file_path, mask_dir_path, feature_path):
    """
    It is a function to obtain a set of numerical features for each nuclei.
    Arguments
        img_file_path: the file path of the input patch.
        mask_dir_path: the file path of segmentation masks predicted by Hover-Net.
        feature_path: the file path for saving extracted nuclei features in the patch.
        barcode: the barcode ID of the input patch.
    """
    im_input = skimage.io.imread(img_file_path)[:, :, :3]
    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin': [0.07, 0.99, 0.11],
        'dab': [0.27, 0.57, 0.78],
        'null': [0.0, 0.0, 0.0]
    }
    # specify stains of input image
    stain_1 = 'hematoxylin'  # nuclei stain
    stain_2 = 'eosin'  # cytoplasm stain
    stain_3 = 'null'  # set to null of input contains only two stains
    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T
    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains

    npy_path = mask_dir_path + '.mat'
    mat_data = sio.loadmat(npy_path)
    im_nuclei_seg_mask = mat_data['inst_map']
    im_nuclei_stain = im_stains[:, :, 0]
    nuclei_features = htk.features.compute_nuclei_features(im_nuclei_seg_mask, im_nuclei_stain)
    nuclei_features["Label"] = nuclei_features["Label"].astype(int)


    nuclei_num = len(nuclei_features["Label"])
    #Only consider patches with more than 3 nuclei
    if nuclei_num >= 3:
        json_path = mask_dir_path.replace('/mat/','/json/') + '.json'
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        type_df = pd.DataFrame.from_dict(json_data['nuc'], orient='index', columns=['type'])
        type_df['Label'] = type_df.index.astype('int')
        type_df = type_df[['Label', 'type']]
        merge_df = pd.merge(type_df, nuclei_features, on="Label")
        merge_df = merge_df.drop(['Identifier.Xmin', 'Identifier.Ymin',
                                  'Identifier.Xmax', 'Identifier.Ymax',
                                  'Identifier.WeightedCentroidX', 'Identifier.WeightedCentroidY'], axis=1)
        merge_df['type'] = merge_df['type'].astype(str)
        merge_df = merge_df[merge_df['type'] != '0']
        merge_df.to_csv(os.path.join(feature_path, os.path.basename(mask_dir_path) + '.csv'), header=True, index=False)


if __name__ == "__main__":

    samples_list = glob.glob("/patches_result/*/*/*")#folder:patches_result/resistant（Classification）/2404992（Patient ID）/1（WSI ID）
    samples_list.sort(reverse=True)
    mask_root_path = '/hovernet/hovernet-seg'
    feature_root_path = '/hovernet/nuclei_features'

    for sample in samples_list:
        feature_path = os.path.join(feature_root_path,sample.split("/")[-3],sample.split("/")[-2],sample.split("/")[-1])
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        files_name = glob.glob(sample + "/*.png")
        for filename in files_name:
            try:

                img_file_path = filename
                mask_dir_path = os.path.join(mask_root_path, sample.split("/")[-3],sample.split("/")[-2],sample.split("/")[-1],'mat',
                                         os.path.splitext(os.path.basename(filename))[0])
                #
                if os.path.exists(os.path.join(feature_path, os.path.basename(mask_dir_path) + '.csv')):
                    continue
                else:
                    extract_seg_features(img_file_path, mask_dir_path, feature_path)

            except:
                print("Error occured in %s" % os.path.join(filename))