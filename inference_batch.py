import sys
import torch
import torch.nn as nn
from spacetimepilot import ModelManager
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import yaml
from types import SimpleNamespace
from spacetimepilot.utils.builder import build_pipeline, build_dataset
from spacetimepilot.utils.misc import save_video, copy_training_files_locally, upload_file_to_s3, dict_to_namespace
import torch.utils.data
from tqdm import tqdm
import lightning as pl


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="config/recammaster_baseline/v2m0_withdoublesrcvideo.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--ckpt_path", "-ckpt",
        type=str,
        default=None,
        help="Override checkpoint path from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    # Add just these 3 specific arguments you asked for
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic"],
        default=None,
        help="Override inference mode from config",
    )
    parser.add_argument(
        "--start_from_last_frame",
        action="store_true",
        help="Override start_from_last_frame setting from config, this is used to test the model support non-identical first frame camera movement, unlike recammaster ",
    )
    args = parser.parse_args()
    return args

class InferenceLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. Load Wan2.1 pre-trained models
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            cfg.models.dit_path,
            cfg.models.text_encoder_path,
            cfg.models.vae_path,
        ])
        
        # 2. Use registration system
        pipeline_cfg = {'type': cfg.pipeline_version}
        pipeline_class = build_pipeline(pipeline_cfg)
        self.pipe = pipeline_class.from_model_manager(model_manager, device="cuda")

        # 4. Load checkpoint
        ckpt_path = cfg.inference.ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.pipe.dit.load_state_dict(state_dict, strict=True)
        self.pipe.to(dtype=torch.bfloat16)

        # Create output directory
        self.output_dir = cfg.inference.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 5. Create a TextVideoCameraDataset instance for frame processing
        self.dataset_processor = build_dataset(dict(
                type=cfg.dataset.type,
                config=cfg
            ))
        
        self.video_idx = cfg.inference.test_videos
        self.cam_idx = cfg.inference.test_cameras

    def test_dataloader(self):
        # For basic mode, use regular camera poses  
        dataset = build_dataset(dict(
            type=self.cfg.dataset.type,
            config=self.cfg              
        ))
        
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.cfg.dataloader.num_workers
        )

    def test_step(self, batch, batch_idx):
        target_text = batch["text"]
        source_video = batch["src_video"]
        src_camera = batch["src_camera"]  
        tgt_camera = batch["tgt_camera"]  
        
        # let's make tgt camera to to first frame of src camera repeat 21 times
        src_time_embedding = batch["src_video_timestamp"]
        tgt_time_embedding = batch["tgt_video_timestamp"]
        source_video = source_video.to(dtype=torch.bfloat16, device=self.device)
        self.pipe.device = self.device
        
        # Use the actual video and camera indices from the batch (not manual calculation)
        actual_video_idx = batch["video_idx"].item() if hasattr(batch["video_idx"], 'item') else batch["video_idx"]
        actual_cam_idx = batch["cam_type"].item() if hasattr(batch["cam_type"], 'item') else batch["cam_type"]

        # Build pipeline call arguments
        pipe_kwargs = {
            "prompt": target_text,
            "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "source_video": source_video,
            "target_camera": tgt_camera,
            "source_camera": src_camera,
            "src_time_embedding": src_time_embedding.to(device=self.device),
            "tgt_time_embedding": tgt_time_embedding.to(device=self.device),
            "cfg_scale": self.cfg.inference.cfg_scale,
            "num_inference_steps": self.cfg.inference.num_inference_steps,
            "seed": self.cfg.inference.seed,
            "tiled": self.cfg.inference.tiled,
        }

        video = self.pipe(**pipe_kwargs)
        # Save video
        filename = f"video_{actual_video_idx:02d}_cam_{actual_cam_idx:02d}.mp4"
        save_video(video, os.path.join(self.output_dir, filename), fps=30, quality=5)

        return {"processed": len(batch["text"])}


if __name__ == '__main__':
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg_dict)
    
    # Override config with command line arguments
    if args.ckpt_path is not None:
        cfg.inference.ckpt_path = args.ckpt_path
    if args.output_dir is not None:
        cfg.inference.output_dir = args.output_dir
    if args.start_from_last_frame is not None:
        # Handle nested structure
        if not hasattr(cfg.inference, 'start_from_last_frame'):
            cfg.inference.start_from_last_frame = SimpleNamespace()
        cfg.inference.start_from_last_frame.basic = args.start_from_last_frame
    
    
    print("#########################"*4)
    print("cfg", cfg)
    print("#########################"*4)
    model = InferenceLightningModule(cfg)
    
    # Create trainer for multi-GPU inference
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy="ddp",
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )
    
    # Run inference
    print(f" Starting inference on {torch.cuda.device_count()} GPUs")
    print(f"📁 Results will be saved to: {cfg.inference.output_dir}")
    print(f"🔧 Using pipeline: {cfg.pipeline_version}")
    print(f"📹 Testing videos: {cfg.inference.test_videos}")
    print(f" Testing cameras: {cfg.inference.test_cameras}")
    print(f" Using camera file: {cfg.inference.camera_file}")
    
    trainer.test(model)
    
    print(f"✅ Inference completed! Check {cfg.inference.output_dir} for results")