import os
import pandas as pd
from glob import glob
import argparse
import cv2
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import cpbd


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
    parser.add_argument('--pd_video_folder',default="", type=str, required=True)
    args = parser.parse_args()

    cpbd_values = []
    video_name_list = os.listdir(args.pd_video_folder)
    for video_idx, video_name in tqdm(video_name_list):
        pd_video_path = os.path.join(args.args.pd_video_folder,video_name)

        assert osp.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        pd_frames = read_mp4(pd_video_path, False, True, False)
        cpbd_value = [cpbd.compute(frame) for frame in tqdm(pd_frames, leave=False)]
        cpbd_values.extend(cpbd_value)
    print('cpbd:', np.array(cpbd_values).mean())
