import os
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage
import numpy as np
from skimage.color import rgb2ycbcr
import torch_fidelity
import tqdm
import lpips
import torch
import json
os.sep.join([''])

def cal_metrics(source_path, output_path='exp/learned_results/temp'):
    metrics_file = os.sep.join([source_path, 'metrics.json'])
    print(metrics_file)
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        if os.path.exists(output_path):
            # 删除文件夹及其内容
            shutil.rmtree(output_path)
        os.makedirs(os.sep.join([output_path, 'orig']), exist_ok=True)
        os.makedirs(os.sep.join([output_path, 'generated']), exist_ok=True)
        n1 = n2 = 0
        for filename in os.listdir(source_path):
            if '_0' in filename and 'orig' not in filename and 'y0' not in filename:
                shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'generated', filename]))
            elif 'orig' in filename:
                shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'orig', filename]))
            else:
                pass

        orig_path = os.sep.join([output_path, 'orig'])
        generated_path = os.sep.join([output_path, 'generated'])
        N = len(os.listdir(generated_path))
        # assert N == 1000
        # Calculated SSIM
        SSIM_sum = 0
        PSNR_sum = 0
        LPIPS_sum = 0
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        # print('calculating PSNR, SSIM & LPIPS')
        with torch.no_grad():
            for k in tqdm.tqdm(range(N)):
                source_path = os.sep.join([orig_path, 'orig_{}.png'.format(k)])
                source_image = skimage.io.imread(source_path)/255.0
                # source_image = rgb2ycbcr(source_image/255.0)[:, :, 0]

                denoise_path = os.sep.join([generated_path, '{}_0.png'.format(k)])
                generated_image = skimage.io.imread(denoise_path)/255.0
                # print(source_image)
                # generated_image = rgb2ycbcr(generated_image/255.0)[:, :, 0]
                # print(source_image.shape)
                # print(generated_image)
                # SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
                SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
                SSIM_sum += SSIM
                PSNR = psnr(source_image, generated_image)
                PSNR_sum += PSNR
                source_image = source_image * 2 - 1
                generated_image = generated_image * 2 - 1
                LPIPS = loss_fn_vgg(torch.tensor(source_image).permute(2,0,1).to(torch.float32).cuda(), torch.tensor(generated_image).permute(2,0,1).to(torch.float32).cuda())
                LPIPS_sum += LPIPS[0,0,0,0]
                # print(SSIM_sum/(k+1))
                # print(PSNR_sum/(k+1))
            # print('Average SSIM: {}'.format(SSIM_sum/N))
            # print('Average LPIPS: {}'.format(LPIPS_sum/N))
            # print('calculating KID & FID')
            Results = torch_fidelity.calculate_metrics(input1=orig_path, input2=generated_path, fid=True)
            # print('PSNR: {:.2f}, SSIM: {:.4f}, LPIPS: {:.4f}, FID: {:.2f}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['frechet_inception_distance']))
            print('{:.2f} | {:.4f} | {:.4f} | {:.2f}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['frechet_inception_distance']))
        metrics = {'PSNR': PSNR_sum/N, 'SSIM': SSIM_sum/N, 'LPIPS': LPIPS_sum.item()/N, 'FID': Results['frechet_inception_distance']}
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        return {'PSNR': PSNR_sum/N, 'SSIM': SSIM_sum/N, 'LPIPS': LPIPS_sum.item()/N, 'FID': Results['frechet_inception_distance']}


if __name__ == '__main__':
    source_path = 'exp/image_samples/ffhq/deblur_nonlinear_noisy/resample/3steps_learned'
    output_path = 'exp/learned_results/imagenet/inpainting_noiseless/ddnm/3steps'
    cal_metrics(source_path=source_path, output_path=output_path)