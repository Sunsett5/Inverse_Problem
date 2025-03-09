# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/3steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/4steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/5steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/7steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/10steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/15steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/20steps --dataset_id 1

CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/3steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/4steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/5steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/7steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/10steps_learned --learned --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/15steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/20steps_learned --learned --dataset_id 1



# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/3steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/4steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/5steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/7steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/10steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/15steps --dataset_id 1
# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/20steps --dataset_id 1

# CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/3steps_learned --learned --dataset_id 1

# 需要继续跑
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/4steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/5steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/7steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/10steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/15steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=7 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/20steps_learned --learned --dataset_id 1

