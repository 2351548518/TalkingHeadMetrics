import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["http_proxy"] = "http://10.13.50.40:7897"
os.environ["https_proxy"] = "http://10.13.50.40:7897"
import glob
import argparse
import cv2

import subprocess
# For FID,PSNR,LMD,CPBD,SSIM,CSIM,LPIPS,V/A-RMSE/SA
from compute_fid import compute_fid
from compute_psnr import compute_psnr
from compute_lmd import compute_lmd
from compute_cpbd import compute_cpbd
from compute_ssim import compute_ssim
from compute_csim import compute_csim
from compute_lpips import compute_lpips
from compute_VArmse import compute_VAmetrics


# get fid ==========================================================
def get_fid(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute fid =======')
    fid = compute_fid(gt_video_folder, pd_video_folder,batch_size=512)
    print(f'[INFO] ======= fid: {fid} =======')
    return fid

# get psnr ==========================================================
def get_psnr(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute psnr =======')
    psnr = compute_psnr(gt_video_folder, pd_video_folder)
    print(f'[INFO] ======= psnr: {psnr} =======')
    return psnr

# get lmd ==========================================================
def get_lmd(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute lmd =======')
    lmd = compute_lmd(gt_video_folder, pd_video_folder)
    print(f'[INFO] ======= lmd: {lmd} =======')
    return lmd

# get cpbd ==========================================================
def get_cpbd(pd_video_folder):
    print(f'[INFO] ======= compute cpbd =======')
    cpbd = compute_cpbd(pd_video_folder)
    print(f'[INFO] ======= cpbd: {cpbd} =======')
    return cpbd

# get ssim ==========================================================
def get_ssim(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute ssim =======')
    ssim = compute_ssim(gt_video_folder, pd_video_folder)
    print(f'[INFO] ======= ssim: {ssim} =======')
    return ssim

# get csim ==========================================================
def get_csim(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute csim =======')
    csim = compute_csim(gt_video_folder, pd_video_folder, batch_size=512, weight='/data2/home/jiapeng2/code/Audio2VideoFace/3DGS/EmoGaussianTalker/data_utils/arcface_torch/backbones/r100.pth')
    print(f'[INFO] ======= csim: {csim} =======')
    return csim

# get lpips ==========================================================
def get_lpips(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute lpips =======')
    lpips = compute_lpips(gt_video_folder, pd_video_folder)
    print(f'[INFO] ======= lpips: {lpips} =======')
    return lpips

# get va ==========================================================
def get_va(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute va =======')
    rmse_valence, rmse_arousal, sagr_valence, sagr_arousal = compute_VAmetrics(gt_video_folder, pd_video_folder,checkpoint_path= "")
    print(f'[INFO] ======= rmse_valence: {rmse_valence} =======')
    print(f'[INFO] ======= rmse_arousal: {rmse_arousal} =======')
    print(f'[INFO] ======= sagr_valence: {sagr_valence} =======')
    print(f'[INFO] ======= sagr_arousal: {sagr_arousal} =======')
    return rmse_valence, rmse_arousal, sagr_valence, sagr_arousal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', default="", type=str, required=True)
    parser.add_argument('--pd_video_folder', default="",type=str, required=True)
    parser.add_argument('--gt_au_folder', default="", type=str, required=True)
    parser.add_argument('--pd_au_folder', default="", type=str, required=True)
    parser.add_argument('--gt_audio_folder', default="", type=str, required=True)
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    args = parser.parse_args()

    # extract audio
    if args.task == -1 or args.task == 1:
        extract_audio(opt.path, wav_path)

    # extract audio features
    if args.task == -1 or args.task == 2:
        extract_audio_features_deepspeech(wav_path)

    # extract images
    if args.task == -1 or args.task == 3:
        extract_images(opt.path, ori_imgs_dir)

    # face parsing
    if args.task == -1 or args.task == 4:
        extract_semantics(ori_imgs_dir, parsing_dir)

    # extract bg
    if args.task == -1 or args.task == 5:
        extract_background(base_dir, ori_imgs_dir)

    # extract torso images and gt_images
    if args.task == -1 or args.task == 6:
        extract_torso_and_gt(base_dir, ori_imgs_dir)
