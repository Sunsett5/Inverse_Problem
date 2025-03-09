# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/3steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/4steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/5steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/7steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/10steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/15steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/20steps

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/15steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddnm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddnm/20steps_learned --learned