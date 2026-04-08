from model import vit_small
import torch
from PIL import Image
from torchvision import transforms
import os
import glob
import argparse
import sys
import pandas as pd




def main():
    parser = argparse.ArgumentParser(description='Compute features from Lunit(VIT/S) embedder')
    parser.add_argument('--dataset', type=str, help='patches folder name')

    parser.add_argument('--output_folder', type=str,
                        help='Features output folder name')
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vit_small(pretrained=True, progress=False, key="DINO_p8", patch_size=8)
    model = model.to(device)
    model = model.eval()
    #The standardized values are sourced from https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean=(0.70322989, 0.53606487, 0.66096631), std=(0.21716536, 0.26081574, 0.20723464))])

    bags_path = os.path.join(args.dataset, '*', '*', "*")#args.dataset/resistant（Classification）/2404992（Patient ID）/1（WSI ID）
    bags_list = glob.glob(bags_path)
    bags_list.sort()
    num_bags = len(bags_list)
    with torch.no_grad():
        for i in range(0, num_bags):
            patches = glob.glob(os.path.join(bags_list[i], '*.png'))
            names = []
            feats_list= []
            for idx, patch in enumerate(patches):
                names.append(patch.split('/')[-1])
                img_ = Image.open(patch)
                img_ = transform(img_).unsqueeze(0)
                img_ = img_.to(device)
                feats_ = model(img_)
                feats_ = feats_.cpu().numpy()
                feats_list.extend(feats_)
                del img_, feats_
                torch.cuda.empty_cache()

            sys.stdout.write('\r Computed: {}/{}'.format(i + 1, num_bags))

            df_ = pd.DataFrame(feats_list)
            df_.index = names
            os.makedirs(os.path.join(args.output_folder, 'lunit_result', bags_list[i].split(os.path.sep)[-3],
                                     bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]),exist_ok=True)
            df_.to_csv(os.path.join(args.output_folder, 'lunit_result', bags_list[i].split(os.path.sep)[-3], bags_list[i].split(os.path.sep)[-2],
                            bags_list[i].split(os.path.sep)[-1], 'patch_features.csv'), float_format='%.4f')
            print('\n')


if __name__ == '__main__':
    main()