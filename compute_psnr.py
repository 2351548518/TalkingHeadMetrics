import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import pandas as pd
from glob import glob
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from piq import psnr


def read_mp4(input_fn, to_rgb=False, to_gray=False, to_nchw=False):
    frames = []
    cap = cv2.VideoCapture(input_fn)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', default="", type=str, required=True)
    parser.add_argument('--pd_video_folder', default="", type=str, required=True)
    args = parser.parse_args()

    psnr_values = []
    video_name_list = os.listdir(args.pd_video_folder)
    for video_idx, video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(args.gt_video_folder,video_name)
        pd_video_path = os.path.join(args.pd_video_folder,video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        gt_frames = read_mp4(gt_video_path, True, False, True)
        pd_frames = read_mp4(pd_video_path, True, False, True)

        gt_frames = torch.from_numpy(gt_frames).float() / 255.
        pd_frames = torch.from_numpy(pd_frames).float() / 255.

        psnr_value = psnr(pd_frames, gt_frames, reduction='none')
        psnr_values.extend([e.item() for e in psnr_value])

    print('psnr:', np.array(psnr_values).mean())
