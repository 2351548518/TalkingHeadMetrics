import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import argparse
import numpy as np
from tqdm import tqdm
import face_alignment
from utils.video_utils import read_mp4


def compute_lmd(gt_video_folder,pd_video_folder):
    lmd_values = []
    lip_range = slice(48, 68)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True)

    video_name_list = os.listdir(pd_video_folder)

    for video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(gt_video_folder,video_name)
        pd_video_path = os.path.join(pd_video_folder,video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        gt_frames = read_mp4(gt_video_path, target_size=None, to_rgb=True, to_gray=False, to_nchw=False)
        pd_frames = read_mp4(pd_video_path, target_size=None, to_rgb=True, to_gray=False, to_nchw=False)
        for gt_frame, pd_frame in tqdm(zip(gt_frames, pd_frames), total=len(gt_frames)):
            gt_landmarks = fa.get_landmarks(gt_frame)
            pd_landmarks = fa.get_landmarks(pd_frame)
            
            if len(gt_landmarks) == 0:
                continue
            gt_landmarks, pd_landmarks = gt_landmarks[0][lip_range, :], pd_landmarks[0][lip_range, :]
            distances = np.abs(pd_landmarks - gt_landmarks)
            lmd_values.append(distances)
    lmd_result = np.array(lmd_values).mean()
    return lmd_result   

if __name__ == '__main__':
    # predict landmark distance only for lips
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', default="", type=str, required=True)
    parser.add_argument('--pd_video_folder', default="", type=str, required=True)
    args = parser.parse_args()

    compute_lmd(args.gt_video_folder, args.pd_video_folder)