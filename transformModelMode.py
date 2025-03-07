from safetensors.torch import load_file, save_file
import torch

# 加载safetensors格式
weights = load_file("/content/Python_sd_web/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors")
# 转换为PyTorch格式（兼容旧代码）
torch.save(weights, "/content/Python_sd_web/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt")
