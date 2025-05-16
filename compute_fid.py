import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import argparse
import numpy as np
import torch
from tqdm import tqdm
from piq import FID
from utils.video_utils import read_mp4

def compute_fid(gt_video_folder, pd_video_folder,batch_size):
    fid_metric = FID()
    gt_feats = []
    pd_feats = []
    video_name_list = os.listdir(pd_video_folder)
    for video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(gt_video_folder,video_name)
        pd_video_path = os.path.join(pd_video_folder,video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        gt_frames = read_mp4(gt_video_path,target_size=None, to_rgb=True, to_gray=False, to_nchw=True)
        pd_frames = read_mp4(pd_video_path,target_size=None, to_rgb=True, to_gray=False, to_nchw=True)

        gt_frames = torch.from_numpy(gt_frames).float() / 255.
        pd_frames = torch.from_numpy(pd_frames).float() / 255.

        T = gt_frames.size(0)
        total_images = torch.cat((gt_frames, pd_frames), 0)
        if len(total_images) > batch_size:
            total_images = torch.split(total_images, batch_size, 0)
        else:
            total_images = [total_images]

        total_feats = []
        for sub_images in total_images:
            feats = fid_metric.compute_feats([
                {'images': sub_images},
            ])
            feats = feats.detach().cpu()
            total_feats.append(feats)
        total_feats = torch.cat(total_feats, 0)
        gt_feat, pd_feat = torch.split(total_feats, (T, T), 0)

        gt_feats.append(gt_feat.numpy())
        pd_feats.append(pd_feat.numpy())

    gt_feats = torch.from_numpy(np.concatenate(gt_feats, 0))
    pd_feats = torch.from_numpy(np.concatenate(pd_feats, 0))
    fid_result = fid_metric.compute_metric(pd_feats, gt_feats).item()
    return fid_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', type=str, required=True)
    parser.add_argument('--pd_video_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    compute_fid(args.gt_video_folder, args.pd_video_folder, args.batch_size)

