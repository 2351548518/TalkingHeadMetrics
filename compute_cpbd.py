import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
from utils.video_utils import read_mp4
import argparse
import numpy as np
from tqdm import tqdm
import cpbd


def compute_cpbd(pd_video_folder):
    cpbd_values = []
    video_name_list = os.listdir(pd_video_folder)
    for video_name in tqdm(video_name_list):
        pd_video_path = os.path.join(pd_video_folder,video_name)

        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        pd_frames = read_mp4(pd_video_path, target_size=None,to_rgb=False, to_gray=True, to_nchw=False)
        cpbd_value = [cpbd.compute(frame) for frame in tqdm(pd_frames, leave=False)]
        cpbd_values.extend(cpbd_value)
    cpbd_results = np.array(cpbd_values).mean()
    return cpbd_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd_video_folder',default="", type=str, required=True)
    args = parser.parse_args()

    compute_cpbd(args.pd_video_folder)
