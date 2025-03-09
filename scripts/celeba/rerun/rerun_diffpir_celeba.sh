
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg deblur_aniso --sigma_0 0.00 -i celeba/deblur_aniso_noiseless/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg deblur_aniso --sigma_0 0.05 -i celeba/deblur_aniso_noisy/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg inpainting --sigma_0 0.00 -i celeba/inpainting_noiseless/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg inpainting --sigma_0 0.05 -i celeba/inpainting_noisy/diffpir/15steps_learned --learned



CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg sr4 --sigma_0 0.00 -i celeba/sr4_noiseless/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg sr4 --sigma_0 0.05 -i celeba/sr4_noisy/diffpir/15steps_learned --learned



CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg cs2 --sigma_0 0.00 -i celeba/cs2_noiseless/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg cs2 --sigma_0 0.05 -i celeba/cs2_noisy/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg deblur_nonlinear --sigma_0 0.00 -i celeba/deblur_nonlinear_noiseless/diffpir/15steps_learned --learned


CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 3 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/3steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 4 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/4steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 5 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/5steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 7 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/7steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 10 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/10steps_learned --learned
CUDA_VISIBLE_DEVICES=2 python main_inverse.py --ni --config celeba_hq.yml --doc celeba --algo diffpir --timesteps 15 --deg deblur_nonlinear --sigma_0 0.05 -i celeba/deblur_nonlinear_noisy/diffpir/15steps_learned --learned
