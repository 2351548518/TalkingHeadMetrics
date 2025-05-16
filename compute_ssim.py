import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import argparse
import numpy as np
import torch
from tqdm import tqdm
from piq import ssim
from utils.video_utils import read_mp4


def compute_ssim(gt_video_folder, pd_video_folder):
    ssim_values = []
    video_name_list = os.listdir(pd_video_folder)
    for video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(gt_video_folder,video_name)
        pd_video_path = os.path.join(pd_video_folder,video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        gt_frames = read_mp4(gt_video_path, target_size=None,to_rgb=True, to_gray=False, to_nchw=True)
        pd_frames = read_mp4(pd_video_path, target_size=None,to_rgb=True, to_gray=False, to_nchw=True)

        gt_frames = torch.from_numpy(gt_frames).float() / 255.
        pd_frames = torch.from_numpy(pd_frames).float() / 255.


        ssim_value = ssim(pd_frames, gt_frames, data_range=1., reduction='none')
        ssim_values.extend([e.item() for e in ssim_value])
    ssim_result = np.array(ssim_values).mean()
    return ssim_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder',default="", type=str, required=True)
    parser.add_argument('--pd_video_folder',default="", type=str, required=True)
    args = parser.parse_args()

    compute_ssim(args.gt_video_folder, args.pd_video_folder)