import os
from huggingface_hub import hf_hub_download

os.makedirs("checkpoints/wan2.1", exist_ok=True)

os.environ.setdefault("HF_HUB_DOWNLOAD_WORKERS", "8")

for filename in [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "diffusion_pytorch_model.safetensors",
]:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        filename=filename,
        local_dir="checkpoints/wan2.1",
    )
