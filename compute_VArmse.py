import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import pandas as pd
from glob import glob
import argparse
import cv2
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import argparse

import torch

from emonet.models import EmoNet
from emonet.metrics import RMSE, SAGR

torch.backends.cudnn.benchmark =  True


def compute_VAmetrics(gt_video_folder, pd_video_folder,checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the model
    print(f'Loading the model from {checkpoint_path}.')
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    net = EmoNet(n_expression=args.nclasses).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()


    video_name_list = os.listdir(gt_video_folder)
    gt_valence_values = []
    gt_arousal_values = []
    pd_valence_values = []
    pd_arousal_values = []
    for video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(gt_video_folder, video_name)
        pd_video_path = os.path.join(pd_video_folder, video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        gt_capture = cv2.VideoCapture(gt_video_path)
        with torch.no_grad():
            # Reads all the frames
            while gt_capture.isOpened():
                ret, frame = gt_capture.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize image to (256,256)
                image_rgb = cv2.resize(image_rgb, (256, 256))
                # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
                image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0

                output = net(image_tensor.unsqueeze(0))
                gt_valence_values.append(output['valence'].clamp(-1.0,1.0).cpu().item())
                gt_arousal_values.append(output['arousal'].clamp(-1.0,1.0).cpu().item())
        gt_capture.release()
        pd_capture = cv2.VideoCapture(pd_video_path)
        with torch.no_grad():
            # Reads all the frames
            while pd_capture.isOpened():
                ret, frame = pd_capture.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize image to (256,256)
                image_rgb = cv2.resize(image_rgb, (256, 256))
                # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
                image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0
                output = net(image_tensor.unsqueeze(0))
                pd_valence_values.append(output['valence'].clamp(-1.0,1.0).cpu().item())
                pd_arousal_values.append(output['arousal'].clamp(-1.0,1.0).cpu().item())
        pd_capture.release()
    gt_valence_values = np.array(gt_valence_values).astype(np.float64)
    gt_arousal_values = np.array(gt_arousal_values).astype(np.float64)
    pd_valence_values = np.array(pd_valence_values).astype(np.float64)
    pd_arousal_values = np.array(pd_arousal_values).astype(np.float64)
    # Compute the metrics
    rmse_valence = RMSE(gt_valence_values, pd_valence_values)
    rmse_arousal = RMSE(gt_arousal_values, pd_arousal_values)

    sagr_valence = SAGR(gt_valence_values, pd_valence_values)
    sagr_arousal = SAGR(gt_arousal_values, pd_arousal_values)
    print('rmse_valence:', rmse_valence)
    print('rmse_arousal:', rmse_arousal)
    print('sagr_valence:', sagr_valence)
    print('sagr_arousal:', sagr_arousal)
    return rmse_valence, rmse_arousal, sagr_valence, sagr_arousal
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder',default="", type=str, required=True)
    parser.add_argument('--pd_video_folder',default="", type=str, required=True)
    parser.add_argument('--checkpoint_path',default="", type=str, required=True)
    args = parser.parse_args()

    # Compute the metrics
    rmse_valence, rmse_arousal, sagr_valence, sagr_arousal = compute_VAmetrics(args.gt_video_folder, args.pd_video_folder, args.checkpoint_path)
