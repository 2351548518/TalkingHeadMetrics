import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用 GPU 这一行需要在import torch前面进行导入，这样才是指定卡
import argparse
import cv2
import lpips
from tqdm import tqdm
import numpy as np
import torch

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return self.measure()

def compute_lpips(gt_video_folder, pd_video_folder):

    lpips_meter = LPIPSMeter()
    lpips_meter.clear()

    video_name_list = os.listdir(pd_video_folder)

    for video_name in tqdm(video_name_list):
        gt_video_path = os.path.join(gt_video_folder,video_name)
        pd_video_path = os.path.join(pd_video_folder,video_name)

        assert os.path.exists(gt_video_path), f"'{gt_video_path}' is not exist"
        assert os.path.exists(pd_video_path), f"'{pd_video_path}' is not exist"

        capture_1 = cv2.VideoCapture(pd_video_path)
        capture_2 = cv2.VideoCapture(gt_video_path)

        while True:
            ret_1, frame_1 = capture_1.read()
            ret_2, frame_2 = capture_2.read()

            if not ret_1 * ret_2:
                break
            
            # plt.imshow(frame_1[:, :, ::-1])
            # plt.show()
            inp_1 = torch.FloatTensor(frame_1[..., ::-1] / 255.0)[None, ...].cuda()
            inp_2 = torch.FloatTensor(frame_2[..., ::-1] / 255.0)[None, ...].cuda()
            lpips_meter.update(inp_1, inp_2)

    lpips_result = lpips_meter.report()
    print('LPIPS:',lpips_result)
    return lpips_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', default="", type=str, required=True)
    parser.add_argument('--pd_video_folder', default="",type=str, required=True)
    args = parser.parse_args()

    compute_lpips(args.gt_video_folder, args.pd_video_folder)



