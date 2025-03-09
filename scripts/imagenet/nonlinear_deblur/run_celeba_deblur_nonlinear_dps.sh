CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/3steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/4steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/5steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/7steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/10steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/15steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/20steps --dataset_id 1

CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/3steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/4steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/5steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/7steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/10steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/15steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.00 -i imagenet/deblur_nonlinear_noiseless/dps/20steps_learned --learned --dataset_id 1



CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/3steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/4steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/5steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/7steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/10steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/15steps --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/20steps --dataset_id 1

CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/3steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/4steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/5steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/7steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/10steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/15steps_learned --learned --dataset_id 1
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config imagenet_256.yml --doc imagenet --algo dps --timesteps 20 --deg deblur_nonlinear --sigma_0 0.05 -i imagenet/deblur_nonlinear_noisy/dps/20steps_learned --learned --dataset_id 1

