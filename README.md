<div align="center" style="font-size: 200%;">

# SSL4SED

[![Documentation](https://img.shields.io/github/actions/workflow/status/wagnerp4/SSL_SED/docs.yml?label=docs&color=3f888f)](https://wagnerp4.github.io/SSL_SED/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3f888f?logo=python&logoColor=3f888f)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/wagnerp4/SSL_SED?color=3f888f)](https://github.com/wagnerp4/SSL_SED/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/wagnerp4/SSL_SED?color=3f888f)](https://github.com/wagnerp4/SSL_SED/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wagnerp4/SSL_SED?color=3f888f)](https://github.com/wagnerp4/SSL_SED/network/members)
[![GitHub Contributors](https://img.shields.io/github/contributors/wagnerp4/SSL_SED?color=3f888f)](https://github.com/wagnerp4/SSL_SED/graphs/contributors)

</div>

# Introduction
This repository was originally based on https://github.com/Audio-WestlakeU/ATST-SED, but will likely change drastically in the future to allow modular integration of new building blocks.

# Get started

(I) DESED & AudioSet:
- **DESED free download for Chinese users**: Downloading the DESED dataset is frustrating, we provide a shared link (shared by Chinese cloud disk) for the [DESED_dataset](https://pan.xunlei.com/s/VNzWiiE1XZGd00jFc_HC72FzA1?pwd=bipt#).
- **Real dataset download**: The 7000+ strongly-labelled audio clips extracted from the AudioSet is provided in [this issue](https://github.com/Audio-WestlakeU/ATST-SED/issues/5).

(II) Environment:
1. Download the pretrained [ATST checkpoint (atst_as2M.ckpt)](https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view?usp=drive_link). Noted that this checkpoint is fine-tuned by the AudioSet-2M.

2. Clone the repo by:

```
git clone https://github.com/wagnerp4/SSL4SED.git
```

3. Dependencies:

(III) Stage 1 Training:

1. Change all required paths in `train/local/confs/stage1.yaml` and `train/local/confs/stage2.yaml` to your own paths. Noted that the pretrained ATST checkpoint path should be changed in **both** files.

2. Start training stage 1 by:

```
python train_stage1.py --gpus YOUR_DEVICE_ID,
```

We also supply a pretrained stage 1 ckpt for you to fine-tune directly. [Stage_1.ckpt](https://drive.google.com/file/d/1_sGve3FySPEqZQKYDO_DVntZ-VWVhtWN/view?usp=drive_link). If you cannot run stage 1 without `accm_grad=1`, we recommend you to use this checkpoint first.

3. When finishing the stage 1 training, change the path of the `model_init` in `train/local/confs/stage2.yaml` to the stage 1 checkpoint path (we saved top-5 models in both stages of training, you could use the best one as the model initialization in the stage 2, but use any one of the top-5 models should give the similar results).

(IV) Stage 2 Training:

```
python train_stage2.py --gpus YOUR_DEVICE_ID,
```