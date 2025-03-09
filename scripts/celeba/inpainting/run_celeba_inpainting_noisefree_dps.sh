# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/3steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/4steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/5steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/7steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/10steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/15steps
# CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/20steps

CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/15steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo dps --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/dps/20steps_learned --learned