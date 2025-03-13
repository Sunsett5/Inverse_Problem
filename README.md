# LLE_inverse_problem
Codebase for paper Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation.

This project is based on:

\- https://github.com/bahjat-kawar/ddrm (DDRM),

\- https://github.com/wyhuai/DDNM (DDNM), and

\- https://github.com/DPS2022/diffusion-posterior-sampling (DPS), and

\- https://github.com/weigerzan/ProjDiff (ProjDiff)
## Environment

You can set up the environment using the `environment.yml` (the requirement is the same as 

[DDRM]: https://github.com/bahjat-kawar/ddrm

). Run

```bash
conda env create -f environment.yml
conda activate LLE
```
Download the nonlinear blurring models from https://github.com/VinAIResearch/blur-kernel-space-exploring.
```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
```

## Experiments in the paper

### Pre-trained models

We use pre-trained models on CelebA-HQ, and FFHQ. Please download the pre-trained models from https://github.com/ermongroup/SDEdit for CelebA-HQ (https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link) and from https://github.com/DPS2022/diffusion-posterior-sampling  for FFHQ (https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh). Place them into exp/logs/celeba and exp/logs/ffhq, respectively.

### Test datasets

We use 1000 test samples for CelebA-HQ and 100 test samples for FFHQ. Download the CelebA-HQ test set from https://github.com/wyhuai/DDNM (https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4) and place it into exp/datasets/celeba_hq. Download the FFHQ test set from https://drive.google.com/file/d/1NcRFk9PPDVoH0F--1KbiGiM_9L79t1Qr/view?usp=drive_link and place it into exp/datasets/ffhq (which is randomly sampled from the folder 00000 of https://github.com/NVlabs/ffhq-dataset). Thus the exp/ folder should look as follows:

```bash
exp
├── logs
├── datasets
│   ├── celeba_hq/celeba_hq # 1000 CelebA-HQ test samples
│   ├── ffhq/00000 # 100 FFHQ test samples
```

### Reproduce the results

Please run the following code:

```
CUDA_VISIBLE_DEVICES=0 python main_inverse.py --ni --config {CONFIG} --doc {DATASET} --algo {ALGO} --timesteps {STEPS} --deg {DEGRADATION} --sigma_0 {SIGMA_0} -i {IMAGE_FOLDER} --learned --default_lr
```

where the following are options

- `ALGO` is the algorithm used. Choose from ddrm, ddnm, pigdm, dps, reddiff, diffpir, resample, dmps, daps
- `STEPS` controls how many timesteps used in the process (we use 3,4,5,7,10,15 in our experiments).
- `DEGREDATION` choose from: `inpainting`, `deblur_aniso`,  `sr4`, `cs2`, `deblur_nonlinear`
- `SIGMA_0` is the noise observed in y (we use 0 and 0.05 in our experiments).
- `CONFIG` is the name of the config file. Choose from `celeba_hq.yml`, and `ffhq.yml`.
- `DATASET` is the name of the dataset used. Choose from `celeba`, and `ffhq`.
- `IMAGE_FOLDER` is the name of the folder the resulting images will be placed in (default: `images`).
- `learned` applies the LLE method.
- `default_lr` provides the default learning rate for reproducing the results in the paper.

e.g., for noise-free inpainting experiment using DDNM with 4 steps on FFHQ with LLE, run

```bash
CUDA_VISIBLE_DEVICES=0 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 4 --deg inpainting --sigma_0 0.00 -i ffhq/inpainting_noiseless/ddnm/4steps_LLE --learned --default_lr
```

To disable LLE, run
```bash
CUDA_VISIBLE_DEVICES=0 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 4 --deg inpainting --sigma_0 0.00 -i ffhq/inpainting_noiseless/ddnm/4steps_original
```

The code will first sample 50 samples using the diffusion model if it's the first time to inference on this dataset.

### Calculate the metrics

We provide the example code for calculating the metrics in `calculate_metrics/`. Please first copy all the images using `mv_files.py` (mainly for FID calculation), and then run `cal_metrics.py` to calculate the metrics. Note that you may need to adjust the paths in the files.

## References and Acknowledgements

