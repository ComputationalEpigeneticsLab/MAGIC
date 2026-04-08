
# Reinhard color normalization for HE-stained histological images using histomicsTK tools.
# Reference: https://digitalslidearchive.github.io/HistomicsTK/examples/nuclei_segmentation.html#Perform-color-normalization
import glob
import os
import PIL
import skimage.io
import skimage.color

import sys
#
histomicstk_path = '/HistomicsTK'
if os.path.exists(histomicstk_path) and histomicstk_path not in sys.path:
    sys.path.insert(0, histomicstk_path)
import histomicstk as htk

def nmzd_reinhard_rescale(input_image_file,nmzd_path):

    im_input = skimage.io.imread(input_image_file)[:, :, :3]
    # Load reference image for normalization
    ref_image_file = '/top_left_coords_46144_9096_.png'
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]
    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)
    # perform reinhard color normalization
    im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)
    pil_img = PIL.Image.fromarray(im_nmzd)
    path = os.path.join(nmzd_path,input_image_file.split("/")[-4],input_image_file.split("/")[-3],input_image_file.split("/")[-2])
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    pil_img.save(os.path.join(path,input_image_file.split("/")[-1]))


samples_list = glob.glob("/patches_result/*/*/*")
nmzd_path = "/hovernet/patches_normalization"
if not os.path.exists(nmzd_path):
        os.makedirs(nmzd_path)

for sample in samples_list:
    files_name = glob.glob(sample+"/*.png")
    for filename in files_name:
        try:
            nmzd_reinhard_rescale(filename,nmzd_path)
        except:
            print(f'Error occured in {filename}')