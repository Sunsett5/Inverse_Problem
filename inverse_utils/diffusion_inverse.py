import os
import logging
import time
import glob

from skimage.metrics import structural_similarity as ssim
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import shutil

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from datasets.npydataset import NpyDataset_cache as NpyDataset
from functions.ckpt_util import get_ckpt_path, download
import lpips

import torchvision.utils as tvu
import torch_fidelity
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random
import yaml
# from ProjDiff_utils.default_lr import get_default_lr
from inverse_utils.sampling import uncond_sampling
from guided_diffusion.unet_ffhq import create_model as create_model_ffhq
from PIL import Image
from torch.nn import Parameter
from algos.ddnm import DDNM
from algos.pigdm import PiGDM
from algos.ddrm import DDRM
from algos.dps import DPS
from algos.reddiff import RED_diff
from algos.diffpir import DiffPIR
from algos.dmps import DMPS
from algos.resample import ReSample
from algos.daps import DAPS


# def 
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.learned = self.args.learned
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = '/nas/datasets/zjw/ddrm/celeba_hq.ckpt'
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'ffhq':
            cls_fn = None
            model_config = load_yaml('configs/ffhq_model_config.yaml')
            model = create_model_ffhq(**model_config)
            model = model.to(self.device)
            model.eval()

        if 'imagenet' in self.args.config:
            dataset_name = 'imagenet'
        elif 'celeba' in self.args.config:
            dataset_name = 'celeba'
        elif 'ffhq' in self.args.config:
            dataset_name = 'ffhq'
        else:
            dataset_name = 'unknown'
        # 先无条件采400个数据出来做训练集，保存在exp/image_samples/trainset
        ## get degradation matrix ##
        deg = self.args.deg
        H_funcs = None
        if 'sr' in deg:
            if deg[:10] == 'sr_bicubic':
                factor = int(deg[10:])
                from obs_functions.Hfuncs import SRConv
                def bicubic_kernel(x, a=-0.5):
                    if abs(x) <= 1:
                        return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                    elif 1 < abs(x) and abs(x) < 2:
                        return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                    else:
                        return 0
                k = np.zeros((factor * 4))
                for i in range(factor * 4):
                    x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                    k[i] = bicubic_kernel(x)
                k = k / np.sum(k)
                kernel = torch.from_numpy(k).float().to(self.device)
                H_funcs = SRConv(kernel / kernel.sum(), \
                                self.config.data.channels, self.config.data.image_size, self.device, stride = factor)
            else:
                # Super-Resolution
                blur_by = int(deg[2:])
                from obs_functions.Hfuncs import SuperResolution
                H_funcs = SuperResolution(self.config.data.channels, self.config.data.image_size, blur_by, self.device)
        elif 'inp' in deg:
            if 'box' in deg:
                missing = torch.zeros([self.config.data.image_size, self.config.data.image_size, self.config.data.channels])
                # left = random.randint(16, 112)
                # up = random.randint(16, 112)
                left = 64
                up = 64
                missing[left:left+128, left:left+128, :] = 1.0
                missing = missing.view(-1).to(self.device).long()
                missing = torch.nonzero(missing).squeeze()  # 去掉多余维度
                print(missing.shape)
            else:
                # Random inpainting
                missing_r = 3 * torch.randperm(self.config.data.image_size**2)[:int(self.config.data.image_size**2 * 0.5)].to(self.device).long()
                missing_g = missing_r + 1
                missing_b = missing_g + 1
                missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            from obs_functions.Hfuncs import Inpainting
            H_funcs = Inpainting(self.config.data.channels, self.config.data.image_size, missing, self.device)
        elif 'deblur_gauss' in deg:
            # Gaussian Deblurring
            from obs_functions.Hfuncs import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), self.config.data.channels, self.config.data.image_size, self.device)
        elif 'phase' in deg:
            # Phase Retrieval
            from obs_functions.Hfuncs import PhaseRetrievalOperator
            H_funcs = PhaseRetrievalOperator(oversample=2.0, device=self.device)
        elif 'hdr' in deg:
            # HDR
            from obs_functions.Hfuncs import HDR
            H_funcs = HDR()   
        elif 'cs' in deg:
            compress_by = int(deg[2:])
            from obs_functions.Hfuncs import WalshHadamardCS
            H_funcs = WalshHadamardCS(self.config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg == 'deblur_aniso':
            from obs_functions.Hfuncs import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), self.config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_nonlinear':
            from obs_functions.Hfuncs import NonlinearBlurOperator
            H_funcs = NonlinearBlurOperator(self.device, opt_yml_path='./bkse/options/generate_blur/default.yml')
        elif deg == 'color':
            from obs_functions.Hfuncs import Colorization
            H_funcs = Colorization(self.config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()

        # for linear observations
        # if 'sr' in deg or 'inp' in deg or 'deblur_gauss' in deg:
        self.args.sigma_0 = 2 * self.args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = self.args.sigma_0
        train_path = '/home/zhangjiawei/scripts/inverse_learned_coeff/exp/image_samples/trainset_{}{}/orig'.format(dataset_name, self.args.dataset_id)
        dst = os.sep.join([self.args.image_folder, 'trainset_{}'.format(dataset_name)])
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        os.symlink(train_path, os.sep.join([dst, 'orig']), target_is_directory=True)
        train_path = dst
        # self.sample_uncond(model, dataset_name, nums=50, cls_fn=cls_fn)
        if self.learned:
            train_epochs = 100
            nums = 50
        else:
            train_epochs = 1
            nums = 1
        if self.args.algo == 'ddnm':
            self.algo = DDNM(model, H_funcs, sigma_0)
        elif self.args.algo == 'pigdm':
            if 'celeba' in self.args.config:
                lam = 1.0
            elif 'imagenet' in self.args.config:
                if 'sr4' in deg:
                    lam = 0.2
                elif 'deblur_aniso' in deg:
                    lam = 0.5
                else:
                    lam = 1.0
            else:
                lam = 1.0
            self.algo = PiGDM(model, H_funcs, sigma_0, lam=lam)
        elif self.args.algo == 'ddrm':
            self.algo = DDRM(model, H_funcs, sigma_0)
        elif self.args.algo == 'dps':
            if 'celeba' in self.args.config:
                if 'cs' in deg:
                    lam = 0.1
                elif deg == 'deblur_nonlinear':
                    lam = 0.1
                elif 'hdr' in deg:
                    lam = 0.01
                elif 'phase' in deg:
                    lam = 0.01
                elif 'deblur_aniso' in deg:
                    lam = 0.5
                elif 'box' in deg:
                    lam = 0.1
                elif 'sr4' in deg:
                    lam = 6.0
                else:
                    lam = 1.0
            elif 'imagenet' in self.args.config:
                if 'inp' in deg:
                    lam = 0.5
                elif 'sr4' in deg:
                    lam = 1.5
                elif 'deblur_aniso' in deg:
                    lam = 0.1
                else:
                    lam = 1.0
            elif 'ffhq' in self.args.config:
                if 'cs' in deg:
                    lam = 0.5
                elif deg == 'deblur_nonlinear':
                    lam = 0.1
                elif 'deblur_aniso' in deg:
                    lam = 0.5
                elif 'inpainting' in deg:
                    lam = 1.0
                elif 'sr4' in deg:
                    lam = 5.0
                else:
                    lam = 1.0
            else:
                lam = 1.0
            self.algo = DPS(model, H_funcs, sigma_0, lam=lam)
        elif self.args.algo == 'reddiff':
            if 'celeba' in self.args.config:
                if 'inp' in deg:
                    if 'box' in deg:
                        eta = 0.4
                    else:
                        eta = 0.5
                elif 'cs' in deg:
                    eta = 0.5
                elif deg == 'deblur_nonlinear':
                    eta = 0.2
                elif 'hdr' in deg:
                    eta = 0.1
                elif 'sr_bicubic' in deg:
                    eta = 3.0
                elif 'sr4' in deg:
                    eta = 7.0
                elif 'deblur_aniso' in deg:
                    eta = 0.5
                else:
                    eta = 1.0
            elif 'imagenet' in self.args.config:
                if 'inp' in deg:
                    eta = 0.5
                elif 'sr4' in deg:
                    eta = 7.0
                elif 'deblur_aniso' in deg:
                    eta = 0.5
                else:
                    eta = 1.0
            elif 'ffhq' in self.args.config:
                if 'cs' in deg:
                    eta = 0.5
                elif deg == 'deblur_nonlinear':
                    eta = 0.2
                elif 'deblur_aniso' in deg:
                    eta = 0.7
                elif 'inpainting' in deg:
                    eta = 0.4
                elif 'sr4' in deg:
                    eta = 7.0
                else:
                    eta = 1.0
            else:
                eta = 1.0
            self.algo = RED_diff(model, H_funcs, sigma_0, eta=eta)
        elif self.args.algo == 'diffpir':
            # if 'deblur_aniso' in deg:
            #     lam = 4.0
            # elif 'cs2' in deg:
            #     lam = 4.0
            # else:
            lam = 7.0
            self.algo = DiffPIR(model, H_funcs, sigma_0, lam=lam)
        elif self.args.algo == 'dmps':
            self.algo = DMPS(model, H_funcs, sigma_0)
        elif self.args.algo == 'resample':
            if deg == 'deblur_aniso':
                gamma = 100
            else:
                gamma = 100
            print(gamma)
            self.algo = ReSample(model, H_funcs, sigma_0, gamma=gamma)
        elif self.args.algo == 'daps':
            self.algo = DAPS(model, H_funcs, sigma_0, betas=self.betas, nonlinear=not H_funcs.is_linear())
        else:
            raise NotImplementedError
        # 再用采样出来的样本你和coeff
        # print(H_funcs.is_linear)
        if H_funcs.is_linear():
            self.train_as_a_whole = False
        else:
            self.train_as_a_whole = True
            print('train as a whole')
        self.coeff_learning(train_path, model, dataset_name, nums, H_funcs, sigma_0, train_epochs)
        self.sample_sequence(train_path, model, cls_fn, H_funcs, sigma_0)


    def coeff_learning(self, train_path, model, dataset_name, nums, H_funcs, sigma_0, train_epochs):
        # 这里需要预先确定最后是多少步的采样
        # train_path = 'exp/image_samples/trainset_{}/'.format(dataset_name)
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps-1, skip)
        seq_next = [-1] + list(seq[:-1])
        step_last = seq[-1]
        eta=0.85
        n = 1
        steps = []
        t_last = 1000
        b = self.betas
        cls_fn = None
        at_last = 0
        last_loss = 1e8
        init_y_0 = True
        from inverse_utils.loss_fn import loss_fn
        if H_funcs.is_linear():
            lam = 0.1
        else:
            lam = 0.1
        loss_fn = loss_fn(lam=lam)
        at_last = torch.tensor([[[[1e-8]]]]).cuda()
        with torch.no_grad():
            for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
                os.makedirs(os.sep.join([train_path, str(i)]), exist_ok=True)
                steps.append(str(i))
                # 读出来所有的数据，过一步DDNM
                for k in range(nums):
                    orig_image = np.load(os.sep.join([train_path, 'orig', '{}.npy'.format(k)]))
                    orig_image = torch.from_numpy(orig_image).cuda().unsqueeze(0)
                    # 做退化
                    if init_y_0:
                        y_0 = H_funcs.H(orig_image)
                        # print(sigma_0)
                        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                        y_0_save = y_0.detach().cpu().numpy()[0]
                        os.makedirs(os.sep.join([train_path, 'y_0']), exist_ok=True)
                        np.save(os.sep.join([train_path, 'y_0', 'y0_{}.npy'.format(k)]), y_0_save)
                    else:
                        y_0 = np.load(os.sep.join([train_path, 'y_0', 'y0_{}.npy'.format(k)]))
                        y_0 = torch.from_numpy(y_0).cuda().unsqueeze(0)

                    t = (torch.ones(n) * i).to(self.device)
                    next_t = (torch.ones(n) * j).to(self.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    # 加噪
                    # print(t_last)
                    if t_last == 1000:
                        xt = torch.randn_like(orig_image)
                        x0_t_last = None
                    else:
                        # 读取数据集中的x0_pred
                        x0_pred_last = np.load(os.sep.join([train_path, str(t_last), 'x0_pred_{}.npy'.format(k)]))
                        x0_pred_last = torch.from_numpy(x0_pred_last).cuda().unsqueeze(0)
                        add_up_last = np.load(os.sep.join([train_path, str(t_last), 'add_up_{}.npy'.format(k)]))
                        add_up_last = torch.from_numpy(add_up_last).cuda().unsqueeze(0)
                        xt = self.algo.map_back(x0_pred_last, y_0, add_up_last, at, at_last)
                        x0_t_last = x0_pred_last
                        # 投影
                    if self.args.algo == 'reddiff':
                        x0_t, add_up = self.algo.cal_x0(xt, x0_t_last, t, at, at_next, y_0, learned=self.learned)
                    else:
                        x0_t, add_up = self.algo.cal_x0(xt, t, at, at_next, y_0, learned=self.learned)
                    # calcultate mu and sigma in DDNM
                    # 保存x0_t_hat
                    add_up_save = add_up.detach().cpu().numpy()[0]
                    x0_t_save = x0_t.detach().cpu().numpy()[0]
                    np.save(os.sep.join([train_path, str(i), 'x0_{}.npy'.format(k)]), x0_t_save)
                    np.save(os.sep.join([train_path, str(i), 'add_up_{}.npy'.format(k)]), add_up_save)
                    # xt_next = x0_t_hat
                    # x0_preds.append(x0_t.to('cpu'))
                    # xs.append(xt_next.to('cpu'))
                # 训练一个可学习的插值向量，分成观测到的部分，和未观测到的部分
                init_y_0 = False
                dataset = NpyDataset(nums=nums, root_dir=train_path, steps=steps)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
                with torch.enable_grad():
                    # lr = max(0.01 * (1-at[0,0,0,0]), 0.0001)
                    # if i == seq[-1]:
                    #     lr = 0.0
                    #     train_epochs = 1
                    # else:
                    if 'imagenet' in self.args.config:
                        if self.args.algo == 'ddnm':
                            if 'inpainting' in self.args.deg:
                                lr = 0.0005 * 20 / self.args.timesteps
                            elif 'sr4' in self.args.deg:
                                lr = 0.0005 * 20 / self.args.timesteps
                            elif 'deblur_aniso' in self.args.deg:
                                lr = 0.0005 * 20 / self.args.timesteps
                        elif self.args.algo == 'dps':
                            lr = 0.005 * 20 / self.args.timesteps
                        elif self.args.algo == 'pigdm':
                            lr = 0.005 * 20 / self.args.timesteps
                        else:
                            lr = 0.002 * 20 / self.args.timesteps
                    # elif 'ffhq' in self.args.config:
                    #     if 'aniso' in self.args.deg:
                    #         if self.args.algo == 'ddnm' or self.args.algo == 'ddrm':
                    #             lr = 0.0005 * 20 / self.args.timesteps
                    elif 'ffhq' in self.args.config:
                        if 'nonlinear' in self.args.deg:
                            if self.args.algo == 'resample' or self.args.algo == 'daps':
                                lr = 0.01 * 20 / self.args.timesteps * at_last[0,0,0,0]
                            else:
                                lr = 0.002 * 20 / self.args.timesteps
                        else:
                            lr = 0.002 * 20 / self.args.timesteps
                    else:
                        lr = 0.002 * 20 / self.args.timesteps
                        # lr = 0.02
                    # lr = 0.5
                    # train_epochs = 100 
                    # lr = 0.01
                    warmup_steps = 50
                    from optim.sf_adamw import AdamWScheduleFree
                    
                    coeff_obs = 0.001 * torch.randn(len(steps))
                    # if len(coeff_obs) > 13:
                    #     coeff_obs[-1] = 1.0 - at_last
                    #     coeff_obs[-2] = at_last
                    # else:
                    #     coeff_obs[-1] = 1.0
                    coeff_obs = Parameter(coeff_obs.cuda().requires_grad_())
                    optimizer_obs = AdamWScheduleFree([coeff_obs], lr=lr, warmup_steps=warmup_steps, foreach=False)
                    coeff_free = 0.001 * torch.randn(len(steps))
                    # if len(coeff_free) > 13:
                    #     coeff_free[-1] = 1.0 - at_last
                    #     coeff_free[-2] = at_last
                    # else:
                    #     coeff_free[-1] = 1.0
                    coeff_free = Parameter(coeff_free.cuda().requires_grad_())
                    optimizer_free = AdamWScheduleFree([coeff_free], lr=lr, warmup_steps=warmup_steps, foreach=False)
                    init = True
                    idx = 0
                    data, gt, y_0 = dataset.get_all_data()
                    data = data.cuda()
                    gt = gt.cuda()
                    y_0 = y_0.cuda()
                    for epoch in range(train_epochs):
                        # for idx, (data, gt, y_0) in enumerate(dataloader):
                        # idx += 1
                        if init:
                            if self.args.algo == 'ddnm' or self.args.algo == 'ddrm':
                                gt = self.algo.get_pred_x(gt, y_0, at_next)
                            init = False
                            with torch.no_grad():
                                loss = loss_fn.get_loss(data[:, -1], gt)['loss']
                                if data.shape[1] > 1:
                                    last_loss = loss_fn.get_loss(data[:, -2], gt)['loss']
                                if loss.item() > last_loss:
                                    if H_funcs.is_linear():
                                    # if False:
                                        coeff_obs[-1] = 0.0
                                        coeff_obs[-2] = 1.0
                                        coeff_free[-1] = 0.0
                                        coeff_free[-2] = 1.0
                                    else:
                                        coeff_obs[-1] = 1.0 - at[0,0,0,0]
                                        coeff_obs[-2] = at[0,0,0,0]
                                        coeff_free[-1] = 1.0 - at[0,0,0,0]
                                        coeff_free[-2] = at[0,0,0,0]
                                else:
                                    coeff_obs[-1] = 1.0
                                    coeff_free[-1] = 1.0 
                        optimizer_obs.zero_grad()
                        optimizer_free.zero_grad()
                        # 先处理一下data，分成观测到的部分和未观测到的部分
                        data_obs = torch.randn_like(data)
                        if self.train_as_a_whole:
                            data_obs = data
                        else:
                            for row in range(data.shape[0]):
                                data_obs[row] = H_funcs.H_pinv(H_funcs.H(data[row])).view(data[row].shape)
                        data_free = data - data_obs
                        # 做拟合
                        pred_obs = None
                        pred_free = None
                        for k_coeff in range(len(steps)):
                            if pred_obs is None:
                                pred_obs = coeff_obs[k_coeff] * data_obs[:, k_coeff]
                                pred_free = coeff_free[k_coeff] * data_free[:, k_coeff]
                            else:
                                pred_obs += coeff_obs[k_coeff] * data_obs[:, k_coeff]
                                pred_free += coeff_free[k_coeff] * data_free[:, k_coeff]
                        pred = pred_obs + pred_free
                        loss_dict = loss_fn.get_loss(pred, gt)
                        loss = loss_dict['loss']
                        loss.backward()
                        
                        optimizer_obs.step()
                        optimizer_free.step()
                        # if self.train_as_a_whole and (epoch + 1) % 100 == 0:
                        #     with torch.no_grad():
                        #         coeff_obs.data = coeff_obs.data - (coeff_obs.data.sum() - 1) / coeff_obs.data.numel()
                        if epoch % 10 == 0 and idx == 0:
                            print('t:{}, epoch:{}, idx:{}/{}, loss_mse:{}, loss_lpips:{}, loss_inception:{}'.format(i, epoch, idx, len(dataloader), loss_dict['loss_mse'].item(), loss_dict['loss_lpips'].item(), loss_dict['loss_inception'].item()))
                    last_loss = loss.item()
                # 把coeff存成npy文件
                np.save(os.sep.join([train_path, str(i), 'coeff_obs.npy']), coeff_obs.detach().cpu().numpy())
                np.save(os.sep.join([train_path, str(i), 'coeff_free.npy']), coeff_free.detach().cpu().numpy())
                # print(coeff_obs)
                # print(coeff_free)
                # 用目前学到的coeff做一遍推理，并存下来
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
                for idx, (data, _, _) in enumerate(dataloader):
                    data = data.cuda()
                    data_obs = torch.randn_like(data)
                    if self.train_as_a_whole:
                        data_obs = data
                    else:
                        for row in range(data.shape[0]):
                            data_obs[row] = H_funcs.H_pinv(H_funcs.H(data[row])).view(data[row].shape)
                    data_free = data - data_obs
                    pred_obs = None
                    pred_free = None
                    for k_coeff in range(len(steps)):
                        if pred_obs is None:
                            pred_obs = coeff_obs[k_coeff] * data_obs[:, k_coeff]
                            pred_free = coeff_free[k_coeff] * data_free[:, k_coeff]
                        else:
                            pred_obs += coeff_obs[k_coeff] * data_obs[:, k_coeff]
                            pred_free += coeff_free[k_coeff] * data_free[:, k_coeff]
                    x0_t_pred = pred_obs + pred_free
                    x0_t_pred_save = x0_t_pred.cpu().numpy()[0]
                    # print(x0_t_pred_save.shape)
                    np.save(os.sep.join([train_path, str(i), 'x0_pred_{}.npy'.format(idx)]), x0_t_pred_save)
                t_last = i
                at_last = at

    def sample_uncond(self, model, dataset_name, nums, cls_fn=None, classes=None):
        os.makedirs('exp/image_samples/trainset_{}/orig/'.format(dataset_name), exist_ok=True)
        with torch.no_grad():
            skip = 1 # 用1000步DDIM采
            seq = range(0, self.num_timesteps-1, skip)
            idx_so_far = 0
            for _ in tqdm.tqdm(range(nums)):
                ##Begin DDIM
                x = torch.randn(
                    1,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )
                x = uncond_sampling(x, seq, model, self.betas, cls_fn=cls_fn, classes=classes)
                x = x[0][-1]
                print(x.shape)

                for j in range(x.size(0)):
                    np.save(os.path.join('exp/image_samples/trainset_{}/orig/'.format(dataset_name), f"{idx_so_far + j}.npy"), x[j].numpy())
                    tvu.save_image(
                        x[j], os.path.join('exp/image_samples/trainset_{}/orig/'.format(dataset_name), f"{idx_so_far + j}.png")
                    )
                idx_so_far += x.size(0)


    def sample_sequence(self, train_path, model, cls_fn=None, H_funcs=None, sigma_0=0.0):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        if 'phase' in args.deg:
            if config.sampling.batch_size > 1:
                key = input('Recommend using batch size 1. Current batch size is {}, switch to 1? [y/n]'.format(config.sampling.batch_size))
                if key == 'y':
                    config.sampling.batch_size = 1
                    print('switch to 1')
                else:
                    print('keep using {}'.format(config.sampling.batch_size))
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        
        # step size
        if args.default_lr: # using default step size to reproduce the metrics
            N = 1
            steps=args.timesteps
            if 'imagenet' in args.config:
                dataset_name = 'imagenet'
            elif 'celeba' in args.config:
                dataset_name = 'celeba'
            elif 'ffhq' in args.config:
                dataset_name = 'ffhq'
            else:
                dataset_name = 'unknown'
            # print(deg)
            # print(steps)
            # print(sigma_0)
            # print(dataset_name)
            lr = get_default_lr(deg, steps, sigma_0, dataset_name)
        else:
            lr = args.lr
            N = args.N

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        with torch.no_grad():
            for x_orig, classes in pbar:
                x_orig = x_orig.to(self.device)
                x_orig = data_transform(self.config, x_orig)

                y_0 = H_funcs.H(x_orig)
                y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                # print(sigma_0)
                y_pinv = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
                # print(y_0.shape)
                for i in range(len(y_0)):
                    tvu.save_image(
                        inverse_data_transform(config, y_pinv[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                    )

                ##Begin DDIM
                x = torch.randn(
                    y_0.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with torch.no_grad():
                    x = self.sample_image(train_path, x, model, H_funcs, y_0, sigma_0, lr, N, last=False, cls_fn=cls_fn, classes=classes)

                x = [inverse_data_transform(config, y) for y in x]

                for j in range(len(x)):
                    tvu.save_image(
                        x[j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x)-1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        # print(torch.norm(orig[0]))
                        mse = torch.mean((x[j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr
                        # print(x[j].shape)
                        avg_ssim += ssim(x[j].numpy(), orig.cpu().numpy(), data_range=x[j].numpy().max() - x[j].numpy().min(), channel_axis=0)
                        LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(x[j]).to(torch.float32).cuda()-1.0)
                        avg_lpips += LPIPS[0,0,0,0]
                idx_so_far += y_0.shape[0]

                pbar.set_description("PSNR:{}, SSIM:{}, LPIPS:{}".format(avg_psnr / (idx_so_far - idx_init), avg_ssim / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init)))

            avg_psnr = avg_psnr / (idx_so_far - idx_init)
            print("Total Average PSNR: %.2f" % avg_psnr)
            print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, train_path, x, model, H_funcs, y_0, sigma_0, lr, N, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps-1, skip)
        print(list(seq))
        seq_next = [-1] + list(seq[:-1])
        xt = x
        n = x.shape[0]
        x0_t_list = None
        b = self.betas
        steps = []
        # print(x.shape)
        x0_t_last = None
        init = True
        with torch.no_grad():
            for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
                steps.append(i)
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                # if init:
                #     init = False
                #     xt = xt * (1-at).sqrt()
                if self.args.algo == 'reddiff':
                    x0_t, add_up = self.algo.cal_x0(xt, x0_t_last, t, at, at_next, y_0, learned=self.learned)
                else:
                    x0_t, add_up = self.algo.cal_x0(xt, t, at, at_next, y_0, learned=self.learned)
                if x0_t_list is None:
                    x0_t_list = x0_t.unsqueeze(1)
                else:
                    x0_t_list = torch.cat((x0_t_list, x0_t.unsqueeze(1)), dim=1)
                # print(x0_t_list.shape)
                x0_t_list_obs = torch.randn_like(x0_t_list)
                if self.train_as_a_whole:
                    x0_t_list_obs = x0_t_list
                else:
                    for row in range(x0_t_list.shape[0]):
                        x0_t_list_obs[row] = H_funcs.H_pinv(H_funcs.H(x0_t_list[row])).view(x0_t_list[row].shape)
                x0_t_list_free = x0_t_list - x0_t_list_obs
                pred_obs = None
                pred_free = None
                coeff_obs = torch.from_numpy(np.load(os.sep.join([train_path, str(i), 'coeff_obs.npy']))).cuda()
                coeff_free = torch.from_numpy(np.load(os.sep.join([train_path, str(i), 'coeff_free.npy']))).cuda()
                if not self.learned:
                    coeff_obs *= 0
                    coeff_free *= 0
                    coeff_obs[-1] = 1.0
                    coeff_free[-1] = 1.0
                for k_coeff in range(len(steps)):
                    if pred_obs is None:
                        pred_obs = coeff_obs[k_coeff] * x0_t_list_obs[:, k_coeff]
                        pred_free = coeff_free[k_coeff] * x0_t_list_free[:, k_coeff]
                    else:
                        pred_obs += coeff_obs[k_coeff] * x0_t_list_obs[:, k_coeff]
                        pred_free += coeff_free[k_coeff] * x0_t_list_free[:, k_coeff]
                x0_t_pred = pred_obs + pred_free
                x0_t_list[:, -1] = x0_t_pred
                x0_t_last = x0_t_pred
                # print(x0_t_list.shape)
                # print(x0_t_pred.shape)
                xt_next = self.algo.map_back(x0_t_pred, y_0, add_up, at_next, at)
                xt = xt_next
        return xt.detach().cpu()