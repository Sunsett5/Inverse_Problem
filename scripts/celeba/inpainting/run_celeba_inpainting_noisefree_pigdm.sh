# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/3steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/4steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/5steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/7steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/10steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/15steps
# CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/20steps

CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/15steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo pigdm --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/pigdm/20steps_learned --learned