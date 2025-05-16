
## **Talking Head Metrics**

There are some Talking Head Metrics code.
## TODO

- [x] 完成代码
- [ ] 完善文档
- [ ] 完成测试
- [ ] 优化

## Installation

1. Clone the repository and set up a conda environment

```
git clone https://github.com/2351548518/TalkingHeadMetrics.git
cd TalkingHeadMetrics
conda create -n talkheadmetrics python=3.8
conda activate talkheadmetrics
pip install -r requirements.txt
```

2. Prepare checkpoints

```
sh download_model.sh
```

For arcface_torch , use the pretrained model: `ms1mv3_arcface_r100_fp16/backbone.pth` , you can find in [arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) , and put it to `weights/arcface_torch`
## Support

| Metrics                                                                                                                               | Paper                                                                                                                                                                                                  | Code Source                                                                                                                                                                                                                                                                                                                | Support      |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| PSNR (peak signal-to-noise ratio)                                                                                                     | -                                                                                                                                                                                                      | [dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)                                                                                                                                                                                                                                      | $\checkmark$ |
| SSIM (structural similarity index measure)                                                                                            | Image quality assessment: from error visibility to structural similarity.                                                                                                                              | [dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)                                                                                                                                                                                                                                      | $\checkmark$ |
| CPBD(cumulative probability of blur detection)                                                                                        | A no-reference image blur metric based on the cumulative probability of blur detection                                                                                                                 | [dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)                                                                                                                                                                                                                                      | $\checkmark$ |
| LPIPS (Learned Perceptual Image Patch Similarity) -                                                                                   | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric                                                                                                                                 | [InsTaG/metrics.py at main · Fictionarry/InsTaG](https://github.com/Fictionarry/InsTaG/blob/main/metrics.py)                                                                                                                                                                                                               | $\checkmark$ |
| FID (Fréchet inception distance)                                                                                                      | GANs trained by a two time-scale update rule converge to a local nash equilibrium                                                                                                                      | [dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline) <br>[mseitzer/pytorch-fid: Compute FID scores with PyTorch.](https://github.com/mseitzer/pytorch-fid)                                                                                                                                | $\checkmark$ |
| LMD (landmark distance error)                                                                                                         | Lip Movements Generation at a Glance                                                                                                                                                                   | [InsTaG/metrics.py at main · Fictionarry/InsTaG](https://github.com/Fictionarry/InsTaG/blob/main/metrics.py)                                                                                                                                                                                                               | $\checkmark$ |
| LSE-D (Lip Sync Error - Distance)<br>LSE-C (Lip Sync Error - Confidence)<br>又叫<br>Sync-D(error distance) <br>Sync-C(confidence score) | Out of time: automated lip sync in the wild                                                                                                                                                            | [Wav2Lip/master 的评估 ·鲁德拉巴/Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)                                                                                                                                                                                                                             | $\checkmark$ |
| CSIM(cosine similarity)                                                                                                               | Arcface: additive angular margin loss for deep face recognition.                                                                                                                                       | [dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)                                                                                                                                                                                                                                      | $\checkmark$ |
| ESD(emotion similarity distance)                                                                                                      | What comprises a good talking-head video generation?: A Survey and Benchmark                                                                                                                           | [talking-head-generation-survey/baseline at master · lelechen63/talking-head-generation-survey](https://github.com/lelechen63/talking-head-generation-survey/tree/master/baseline)                                                                                                                                         | $\times$     |
| AED(average expression distance)<br>APD(average pose distance)                                                                        | Accurate 3d face reconstruction with weakly-supervised learning                                                                                                                                        | [vico_challenge_baseline/Deep3DFaceRecon_pytorch at main · dc3ea9f/vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline/tree/main/Deep3DFaceRecon_pytorch)                                                                                                                                          | $\times$     |
| AKD(average keypoint distance)                                                                                                        | How far are we from solving the 2D & 3D Face Alignment problem?                                                                                                                                        | [1adrianb/face-alignment： ：fire： 使用 pytorch 构建 2D 和 3D 人脸对齐库](https://github.com/1adrianb/face-alignment)                                                                                                                                                                                                                  | $\times$     |
| AUE-U(upper-face action unit error)<br>AUE-L(lower-face action unit error)                                                            | AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis<br>TalkingGaussian: Structure-Persistent 3D Talking  Head Synthesis via Gaussian Splatting<br>AD-NeRF首先提出，TalkingGaussian放出了代码 | [TalkingGaussian/auerror.py at main · Fictionarry/TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian/blob/main/auerror.py)                                                                                                                                                                                    | $\checkmark$ |
| V-RMSE and A-RMSE(valance and arousal root mean square error)<br>V-SAGR and A-SAGR(valance and arousal sign agreement)                | EmoTalkingGaussian: Continuous Emotion-conditioned Talking Head Synthesis<br>EmoTalkingGaussian首次提出，用来评估连续表情VA                                                                                         | [face-analysis/emonet: Official implementation of the paper "Estimation of continuous valence and arousal levels from faces in naturalistic conditions", Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, Nature Machine Intelligence, 2021](https://github.com/face-analysis/emonet) | $\checkmark$ |

## Data

文件目录应该这样

```
data/
├── gt_au_csv
├── gt_audio
├── gt_video
├── pred_au_csv
└── pred_video
```
## Usage

```
# For FID,PSNR,LMD,CPBD,SSIM,CSIM,LPIPS,V/A-RMSE/SA,AUE-L/U

# FOR LIP-SYNC

```
## Acknowledgement

This code is developed on [vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)  , [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian) and [emonet](https://github.com/face-analysis/emonet). Thanks for these great projects!