# SSL4SED

Reference Repository: https://github.com/Audio-WestlakeU/ATST-SED

## Data and Environment

**DESED & AudioSet:**
- **DESED free download for Chinese users**: Downloading the DESED dataset is frustrating, the authors of ATST-SED provide a shared link (shared by Chinese cloud disk) for the [DESED_dataset](https://pan.xunlei.com/s/VNzWiiE1XZGd00jFc_HC72FzA1?pwd=bipt#).
- **Real dataset download**: The 7000+ strongly-labelled audio clips extracted from the AudioSet is provided in [this issue](https://github.com/Audio-WestlakeU/ATST-SED/issues/5).
- TODO: Plans to include links to: MAESTRO, dcase subsets, UrbanSED, TUT-2017-SED

**Environment:**
1. Download the pretrained [ATST checkpoint (atst_as2M.ckpt)](https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view?usp=drive_link). Noted that this checkpoint is fine-tuned by the AudioSet-2M.

2. Clone the repository with `git clone https://github.com/wagnerp4/SSL4SED.git`

3. Setup environment and install dependencies:
```bash
python3 -m venv .venv # uv venv
source .venv/bin/activate  # .venv\Scripts\activate (windows)
pip install -e .
# alternatively, dev mode deps:
pip install -e ".[dev]"
```

## Training (Stage-1)

1. Change all required paths in `train/local/confs/stage1.yaml` and `train/local/confs/stage2.yaml` to your own paths. Noted that the pretrained ATST checkpoint path should be changed in **both** files.

2. Start training stage 1 by:

```
python train_stage1.py --gpus YOUR_DEVICE_ID,
```

The authors of ATST-SED also supply a pretrained stage 1 ckpt for you to fine-tune directly. [Stage_1.ckpt](https://drive.google.com/file/d/1_sGve3FySPEqZQKYDO_DVntZ-VWVhtWN/view?usp=drive_link). If you cannot run stage 1 without `accm_grad=1`, we recommend you to use this checkpoint first.

3. When finishing the stage 1 training, change the path of the `model_init` in `train/local/confs/stage2.yaml` to the stage 1 checkpoint path (the authors of ATST-SED saved top-5 models in both stages of training, you could use the best one as the model initialization in the stage 2, but use any one of the top-5 models should give the similar results).

## Training (Stage-2)
```
python train_stage2.py --gpus YOUR_DEVICE_ID,
```

## Support development with laptop:
```bash
python src/training/train_stage1.py --fast_dev_run --subset_fraction 0.1 # stage 1: ~1M trainable params
python src/training/train_stage2.py --fast_dev_run --subset_fraction 0.01 # stage 2 ~170M trainable params
```

