CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/15steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/20steps

CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/15steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/resample/20steps_learned --learned



CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/15steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/20steps

CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/15steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/resample/20steps_learned --learned



# daps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/3steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/4steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/5steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/7steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/10steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/15steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/20steps

CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 3 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/3steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 4 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/4steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 5 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/5steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 7 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/7steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 10 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/10steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 15 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/15steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 20 --deg inpainting --sigma_0 0.00 -i inpainting_noiseless/daps/20steps_learned --learned



CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 3 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/3steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 4 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/4steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 5 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/5steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 7 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/7steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 10 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/10steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 15 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/15steps
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 20 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/20steps

CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 3 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/3steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 4 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/4steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 5 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/5steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 7 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/7steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 10 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/10steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 15 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/15steps_learned --learned
CUDA_VISIBLE_DEVICES=3 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo daps --timesteps 20 --deg inpainting --sigma_0 0.05 -i inpainting_noisy/daps/20steps_learned --learned