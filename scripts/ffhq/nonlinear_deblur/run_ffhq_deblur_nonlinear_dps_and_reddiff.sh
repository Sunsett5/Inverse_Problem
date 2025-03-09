CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/3steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/4steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/5steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/7steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/10steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/15steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/20steps

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/15steps_learned --learned
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/dps/20steps_learned --learned



CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/3steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/4steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/5steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/7steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/10steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/15steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/20steps

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/15steps_learned --learned
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/dps/20steps_learned --learned





CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/3steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/4steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/5steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/7steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/10steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/15steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/20steps

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/15steps_learned --learned
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i ffhq/deblur_nonlinear_noiseless/reddiff/20steps_learned --learned



CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/3steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/4steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/5steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/7steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/10steps
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/15steps
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/20steps

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/15steps_learned --learned
# CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo reddiff --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i ffhq/deblur_nonlinear_noisy/reddiff/20steps_learned --learned

