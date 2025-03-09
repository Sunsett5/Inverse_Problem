CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 3 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 4 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 5 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 7 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 10 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 15 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddnm/15steps_learned --learned

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 3 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 4 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 5 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 7 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 10 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 15 --deg inpainting --sigma_0 0.05 -i ffhq/inpainting_noisy/ddrm/15steps_learned --learned



CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 3 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 4 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 5 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 7 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 10 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 15 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddnm/15steps_learned --learned

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 3 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 4 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 5 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 7 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 10 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 15 --deg sr4 --sigma_0 0.05 -i ffhq/sr4_noisy/ddrm/15steps_learned --learned



CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 3 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 4 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 5 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 7 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 10 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddnm --timesteps 15 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddnm/15steps_learned --learned

CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 3 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/3steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 4 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/4steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 5 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/5steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 7 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/7steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 10 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/10steps_learned --learned
CUDA_VISIBLE_DEVICES=1 python main_inverse.py --ni --config ffhq.yml --doc ffhq --algo ddrm --timesteps 15 --deg deblur_aniso --sigma_0 0.05 -i ffhq/deblur_aniso_noisy/ddrm/15steps_learned --learned