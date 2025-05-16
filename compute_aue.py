'''
Adapted from https://github.com/Fictionarry/TalkingGaussian/blob/main/auerror.py

The AUE evaluation is based on OpenFace (https://github.com/TadasBaltrusaitis/OpenFace). 
First, use OpenFace's FeatureExtraction to process the reconstructed video ("A_generated.mp4" for example) 
and the corresponding GT ("A_GT.mp4" for example) respectively.
'''

import argparse
import pandas as pd
import os
from tqdm import tqdm


def compute_aue(gt_au_folder, pd_au_folder):
     AUitems = ['AU01_r','AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
     AUitems_lower = ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
     AUitems_upper = ['AU01_r','AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU45_r']

     csv_name_list = os.listdir(gt_au_folder)

     gt_au_values = pd.DataFrame(columns=AUitems)
     pd_au_values = pd.DataFrame(columns=AUitems)

     for csv_name in tqdm(csv_name_list):
         gt_au_path = os.path.join(gt_au_folder, csv_name)
         pd_au_path = os.path.join(pd_au_folder, csv_name)

         assert os.path.exists(gt_au_path), f"'{gt_au_path}' is not exist"
         assert os.path.exists(pd_au_path), f"'{pd_au_path}' is not exist"

         gt_df = pd.read_csv(gt_au_path)[AUitems]
         pd_df = pd.read_csv(pd_au_path)[AUitems]
         gt_au_values = pd.concat([gt_au_values, gt_df], axis=0)
         pd_au_values = pd.concat([pd_au_values, pd_df], axis=0)

     error_l = (gt_au_values[AUitems_lower] - pd_au_values[AUitems_lower]) ** 2
     error_u = (gt_au_values[AUitems_upper] - pd_au_values[AUitems_upper]) ** 2
     aue_l = error_l.mean().sum()
     aue_u = error_u.mean().sum()
     return aue_l, aue_u


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_au_folder',default="", type=str, required=True)
    parser.add_argument('--pd_au_folder',default="", type=str, required=True)
    args = parser.parse_args()

    compute_aue(args.gt_au_folder, args.pd_au_folder)
    