source /content/Python_sd_web/venv/bin/activate
# python txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt /content/Python_sd_web/repositories/stable-diffusion-stability-ai/trinart2_step115000.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 768 --W 768  

python txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --device cuda --ckpt /content/Python_sd_web/repositories/stable-diffusion-stability-ai/trinart2_step115000.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 768 --W 768  