CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/resample/15steps_learned --learned

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/resample/15steps_learned --learned



CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/resample/15steps_learned --learned

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/resample/15steps_learned --learned




CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/resample/15steps_learned --learned

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/resample/15steps_learned --learned


CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/resample/15steps_learned --learned

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/resample/15steps_learned --learned



CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/resample/15steps_learned --learned

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/3steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/4steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/5steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/7steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/10steps
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/15steps

CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/3steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/4steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/5steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/7steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/10steps_learned --learned
CUDA_VISIBLE_DEVICES=4 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo resample --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/resample/15steps_learned --learned