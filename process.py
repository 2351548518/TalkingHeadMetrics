import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["http_proxy"] = "http://10.13.50.40:7897"
os.environ["https_proxy"] = "http://10.13.50.40:7897"
import argparse
import subprocess
import time
# For FID,PSNR,LMD,CPBD,SSIM,CSIM,LPIPS,V/A-RMSE/SA
from compute_fid import compute_fid
from compute_psnr import compute_psnr
from compute_lmd import compute_lmd
from compute_cpbd import compute_cpbd
from compute_ssim import compute_ssim
from compute_csim import compute_csim
from compute_lpips import compute_lpips
from compute_VArmse import compute_VAmetrics
from compute_aue import compute_aue


# get fid ==========================================================
def get_fid(gt_video_folder, pd_video_folder,batch_size):
    print(f'[INFO] ======= compute fid =======')
    fid = compute_fid(gt_video_folder, pd_video_folder,batch_size)
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
def get_csim(gt_video_folder, pd_video_folder,batch_size,weight):
    print(f'[INFO] ======= compute csim =======')
    csim = compute_csim(gt_video_folder, pd_video_folder, batch_size, weight)
    print(f'[INFO] ======= csim: {csim} =======')
    return csim

# get lpips ==========================================================
def get_lpips(gt_video_folder, pd_video_folder):
    print(f'[INFO] ======= compute lpips =======')
    lpips = compute_lpips(gt_video_folder, pd_video_folder)
    print(f'[INFO] ======= lpips: {lpips} =======')
    return lpips

# get va ==========================================================
def get_va(gt_video_folder, pd_video_folder,checkpoint_path):
    print(f'[INFO] ======= compute va =======')
    rmse_valence, rmse_arousal, sagr_valence, sagr_arousal = compute_VAmetrics(gt_video_folder, pd_video_folder,checkpoint_path)
    print(f'[INFO] ======= rmse_valence: {rmse_valence} =======')
    print(f'[INFO] ======= rmse_arousal: {rmse_arousal} =======')
    print(f'[INFO] ======= sagr_valence: {sagr_valence} =======')
    print(f'[INFO] ======= sagr_arousal: {sagr_arousal} =======')
    return rmse_valence, rmse_arousal, sagr_valence, sagr_arousal

# get aue ==========================================================
def get_aue( gt_au_folder, pd_au_folder):
    print(f'[INFO] ======= compute aue =======')
    aue_l,aue_u = compute_aue(gt_au_folder, pd_au_folder)
    print(f'[INFO] ======= aue_l: {aue_l} =======')
    print(f'[INFO] ======= aue_u: {aue_u} =======')
    return aue_l,aue_u


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', default="", type=str, required=True)
    parser.add_argument('--pd_video_folder', default="",type=str, required=True)
    parser.add_argument('--gt_au_folder', default="", type=str, required=True)
    parser.add_argument('--pd_au_folder', default="", type=str, required=True)
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    args = parser.parse_args()

    # get fid
    if args.task == -1 or args.task == 1:
        fid_result = get_fid(args.gt_video_folder, args.pd_video_folder,batch_size=512)

    # get psnr
    if args.task == -1 or args.task == 2:
        psnr_result = get_psnr(args.gt_video_folder, args.pd_video_folder)

    # get lmd
    if args.task == -1 or args.task == 3:
        lmd_result = get_lmd(args.gt_video_folder, args.pd_video_folder)

    # get cpbd
    if args.task == -1 or args.task == 4:
        cpbd_result = get_cpbd(args.pd_video_folder)

    # get ssim
    if args.task == -1 or args.task == 5:
        ssim_result = get_ssim(args.gt_video_folder, args.pd_video_folder)

    # get csim
    if args.task == -1 or args.task == 6:
        csim_result = get_csim(args.gt_video_folder, args.pd_video_folder,batch_size=512, weight='/data2/home/jiapeng2/code/Audio2VideoFace/3DGS/EmoGaussianTalker/data_utils/arcface_torch/backbones/r100.pth')

    # get lpips
    if args.task == -1 or args.task == 7:
        lpips_result = get_lpips(args.gt_video_folder, args.pd_video_folder)
    
    # get va
    if args.task == -1 or args.task == 8:
        rmse_valence, rmse_arousal, sagr_valence, sagr_arousal = get_va(args.gt_video_folder, args.pd_video_folder,checkpoint_path = "")

    # get aue
    if args.task == -1 or args.task == 9:
        aue_l,aue_u = get_aue(args.gt_au_folder, args.pd_au_folder)

    # write to file, the file name is {data}_metrics.txt
    current_time = time.localtime()
    formatted_time = time.strftime('%Y_%m_%d_%H_%M_%S', current_time)
    file_name = f"metrics_{formatted_time}.txt"
    if args.task == -1:
        with open(file_name, 'w') as f:
            f.write(f'fid: {fid_result}\n')
            f.write(f'psnr: {psnr_result}\n')
            f.write(f'lmd: {lmd_result}\n')
            f.write(f'cpbd: {cpbd_result}\n')
            f.write(f'ssim: {ssim_result}\n')
            f.write(f'csim: {csim_result}\n')
            f.write(f'lpips: {lpips_result}\n')
            f.write(f'rmse_valence: {rmse_valence}\n')
            f.write(f'rmse_arousal: {rmse_arousal}\n')
            f.write(f'sagr_valence: {sagr_valence}\n')
            f.write(f'sagr_arousal: {sagr_arousal}\n')
            f.write(f'aue_l: {aue_l}\n')
            f.write(f'aue_u: {aue_u}\n')
    