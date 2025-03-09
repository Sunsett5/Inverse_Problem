# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/3steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/4steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/5steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/7steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/10steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/15steps
# CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/20steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/15steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo ddrm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/ddrm/20steps_learned --learned