"""
Single-video inference for SpaceTimePilot.

Usage:
    python single_video_test.py \
        --video_path data_for_evaluation/exam_61/videos/video_0.mp4 \
        --caption "A crowd on a street" \
        --temporal_control pingpong \
        --cam_type cam01 \
        --src_vid_cam data_for_evaluation/exam_61/src_cam/video_0_extrinsics.npy \
        --ckpt checkpoints/SpacetimePilot_1.3B_v1.ckpt \
        --output_dir ./results/single_test

Time modes:
    forward, reverse, pingpong,
    freeze_start (frame 0), freeze_early (frame 20), freeze_mid (frame 40), freeze_late (frame 60), freeze_end (frame 80),
    bounce_late, bounce_early, slowmo_first_half, slowmo_second_half,
    ramp_then_freeze

Camera type:
    Integer 1–10 selecting a trajectory from the camera_file JSON.
    1=Pan Right, 2=Pan Left, 3=Tilt Up, 4=Tilt Down, 5=Zoom In,
    6=Zoom Out, 7=Translate Up, 8=Translate Down, 9=Arc Left, 10=Arc Right
    Use --camera_file to point to your own camera_extrinsics.json;
    defaults to demo_videos/cameras/camera_extrinsics.json.
"""

import os
import json
import argparse
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import v2

from spacetimepilot import ModelManager
from spacetimepilot.utils.builder import build_pipeline
from spacetimepilot.utils.misc import save_video
from spacetimepilot.dataset.utils import (
    load_frames_using_imageio,
    process_camera_trajectory,
    compute_pose_embedding,
)

# ── Default paths ────────────────────────────────────────────────────────────

DEFAULT_DIT_PATH  = "checkpoints/wan2.1/diffusion_pytorch_model.safetensors"
DEFAULT_TEXT_PATH = "checkpoints/wan2.1/models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE_PATH  = "checkpoints/wan2.1/Wan2.1_VAE.pth"
DEFAULT_CAMERA_FILE = "demo_videos/cameras/camera_extrinsics.json"

PIPELINE_VERSION = "spacetimepilot_1dconv"

# ── Frame / video constants ───────────────────────────────────────────────────

NUM_FRAMES = 81
HEIGHT = 480
WIDTH  = 832

FRAME_PROCESS = v2.Compose([
    v2.CenterCrop(size=(HEIGHT, WIDTH)),
    v2.Resize(size=(HEIGHT, WIDTH), antialias=True),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

VALID_TIME_MODES = {
    "forward", "reverse", "pingpong",
    "bounce_late", "bounce_early",
    "slowmo_first_half", "slowmo_second_half",
    "ramp_then_freeze",
    "freeze_start", "freeze_early", "freeze_mid", "freeze_late", "freeze_end",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_time_pattern(pattern: str, T: int = 81):
    """Return a list of T frame indices for the given temporal pattern."""
    if pattern == "reverse":
        base = list(range(T - 1, -1, -1))
    elif pattern == "pingpong":
        s = 40
        base = list(range(s, T)) + list(range(T - 1, s - 1, -1))
    elif pattern == "bounce_late":
        fa, fb, fc = 4 * 15, 4 * 21, 4 * 5
        base = list(range(fa, fb + 1)) + list(range(fb, fc - 1, -1))
    elif pattern == "bounce_early":
        fa, fb, fc = 4 * 5, 4 * 21, 4 * 15
        base = list(range(fa, fb + 1)) + list(range(fb, fc - 1, -1))
    elif pattern == "slowmo_first_half":
        base = [0] + [i for i in range(1, 41) for _ in (0, 1)]
    elif pattern == "slowmo_second_half":
        base = [40] + [i for i in range(41, T) for _ in (0, 1)]
    elif pattern == "ramp_then_freeze":
        freeze = 40
        base = list(range(freeze + 1)) + [freeze] * (T - freeze - 1)
    elif pattern == "forward":
        base = list(range(T))
    elif pattern == "freeze_mid":
        base = [40.0] * T
    elif pattern == "freeze_early":
        base = [20.0] * T
    elif pattern == "freeze_start":
        base = [0.0] * T
    elif pattern == "freeze_end":
        base = [80.0] * T
    elif pattern == "freeze_late":
        base = [60.0] * T
    else:
        raise ValueError(f"Unknown time pattern: {pattern!r}")

    if len(base) >= T:
        return base[:T]
    # extend by cycling if short
    out, i = [], 0
    while len(out) < T:
        out.append(base[i % len(base)])
        i += 1
    return out


def load_src_camera(raw_w2c: np.ndarray):
    """
    Convert an (81, 4, 4) array of w2c matrices to a (21, 12) bfloat16 tensor.
    Normalises poses relative to the first frame; handles static cameras.
    """
    src_c2w = np.linalg.inv(raw_w2c)          # (81, 4, 4)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = src_c2w @ ref_inv           # (81, 4, 4) relative to first

    translations = src_c2w_norm[:, :3, 3]
    scene_scale  = np.max(np.abs(translations))
    if scene_scale < 1e-2:
        scene_scale = 1.0
    src_c2w_norm[:, :3, 3] /= scene_scale

    src_c2w_norm = src_c2w_norm[::4]           # (21, 4, 4)
    poses = [torch.as_tensor(src_c2w_norm[i])[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)        # (21, 3, 4)
    src_cam = rearrange(src_cam, 'b c d -> b (c d)')  # (21, 12)
    return src_cam.to(torch.bfloat16)


def make_identity_src_camera():
    """Return a static (identity) source camera as (21, 12) bfloat16 tensor."""
    identity_w2c = np.eye(4)[np.newaxis].repeat(81, axis=0)   # (81, 4, 4)
    return load_src_camera(identity_w2c)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpaceTimePilot single-video inference")

    # Required inputs
    p.add_argument("--video_path",        required=True, help="Path to input video file")
    p.add_argument("--caption",           required=True, help="Text description of the video")
    p.add_argument("--temporal_control",  default="forward",
                   choices=sorted(VALID_TIME_MODES),
                   help="Temporal control pattern")
    p.add_argument("--cam_type",          default="cam01",
                   help="Camera trajectory key (e.g. cam01–cam10) from the camera_file JSON")

    # Camera files
    p.add_argument("--camera_file",  default=DEFAULT_CAMERA_FILE,
                   help="Path to camera_extrinsics.json")
    p.add_argument("--src_vid_cam",  default=None,
                   help="Path to source camera .npy file (81×4×4 w2c matrices). "
                        "If omitted, a static identity camera is used.")

    # Model / checkpoint
    p.add_argument("--ckpt",         required=True, help="Path to model checkpoint (.ckpt)")
    p.add_argument("--output_dir",   default="./results/single_test",
                   help="Directory to save the generated video")
    p.add_argument("--dit_path",     default=DEFAULT_DIT_PATH)
    p.add_argument("--text_encoder_path", default=DEFAULT_TEXT_PATH)
    p.add_argument("--vae_path",     default=DEFAULT_VAE_PATH)

    # Inference hyper-params
    p.add_argument("--cfg_scale",          type=float, default=5.0)
    p.add_argument("--num_inference_steps",type=int,   default=20)
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--tiled",              action="store_true", default=True,
                   help="Enable tiled inference (saves VRAM)")

    return p.parse_args()


def _parse_cam_type(cam_type_str: str) -> int:
    """Accept 'cam01', 'cam1', or plain '1' and return the integer index."""
    s = cam_type_str.lower().lstrip("cam").lstrip("0") or "0"
    return int(s)


def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalise cam_type to integer (e.g. "cam01" → 1)
    cam_idx = _parse_cam_type(args.cam_type)

    # ── 1. Load models ─────────────────────────────────────────────────────
    print("Loading Wan2.1 foundation models...")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([args.dit_path, args.text_encoder_path, args.vae_path])

    pipeline_class = build_pipeline({"type": PIPELINE_VERSION})
    pipe = pipeline_class.from_model_manager(model_manager, device="cuda")

    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to(dtype=torch.bfloat16)
    # Move the full DiT to CUDA before enable_vram_management so that custom
    # modules (temporal_downsampler, cam_encoder, cross_attn_cam, etc.) that
    # are not wrapped by the vram manager end up on GPU.
    pipe.dit.to("cuda")
    pipe.device = torch.device("cuda")
    pipe.enable_vram_management()

    # ── 2. Load video ──────────────────────────────────────────────────────
    print(f"Loading video: {args.video_path}")
    video = load_frames_using_imageio(
        args.video_path,
        max_num_frames=NUM_FRAMES,
        start_frame_id=0,
        interval=1,
        num_frames=NUM_FRAMES,
        frame_process=FRAME_PROCESS,
        target_width=WIDTH,
        target_height=HEIGHT,
    )
    if video is None:
        raise ValueError(f"Could not load video: {args.video_path}")
    # video shape: (C, T, H, W) → add batch dim
    source_video = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")  # (1, C, T, H, W)

    # ── 3. Source camera ───────────────────────────────────────────────────
    if args.src_vid_cam is not None:
        print(f"Loading source camera: {args.src_vid_cam}")
        raw_w2c = np.load(args.src_vid_cam)   # (81, 4, 4)
        src_cam = load_src_camera(raw_w2c)
    else:
        print("No source camera provided; using static identity camera.")
        src_cam = make_identity_src_camera()
    # Add batch dim: (1, 21, 12)
    src_camera = src_cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")

    # ── 4. Target camera ───────────────────────────────────────────────────
    print(f"Loading target camera: {args.cam_type} (index {cam_idx}), file={args.camera_file}")
    with open(args.camera_file, "r") as f:
        cam_data = json.load(f)

    frame_indices = list(range(NUM_FRAMES))[::4]   # [0, 4, 8, …, 80] → 21 frames
    tgt_cam_params = process_camera_trajectory(cam_data, frame_indices, cam_idx)
    tgt_cam = compute_pose_embedding(tgt_cam_params)  # (21, 3, 4), bfloat16
    tgt_cam = rearrange(tgt_cam, 'b c d -> b (c d)')   # (21, 12)
    tgt_camera = tgt_cam.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")  # (1, 21, 12)

    # ── 5. Time embeddings ─────────────────────────────────────────────────
    print(f"Temporal control: {args.temporal_control}")
    src_indices = get_time_pattern("forward",             NUM_FRAMES)
    tgt_indices = get_time_pattern(args.temporal_control, NUM_FRAMES)

    src_time = torch.tensor(src_indices, dtype=torch.float32).unsqueeze(0).to("cuda")  # (1, 81)
    tgt_time = torch.tensor(tgt_indices, dtype=torch.float32).unsqueeze(0).to("cuda")  # (1, 81)

    # ── 6. Run pipeline ────────────────────────────────────────────────────
    NEGATIVE_PROMPT = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    print("Running inference...")
    video_out = pipe(
        prompt=args.caption,
        negative_prompt=NEGATIVE_PROMPT,
        source_video=source_video,
        target_camera=tgt_camera,
        source_camera=src_camera,
        src_time_embedding=src_time,
        tgt_time_embedding=tgt_time,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=args.tiled,
    )

    # ── 7. Save result ─────────────────────────────────────────────────────
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    out_name = f"{video_name}_time{args.temporal_control}_{args.cam_type}.mp4"
    out_path = os.path.join(args.output_dir, out_name)
    save_video(video_out, out_path, fps=30, quality=5,
               ffmpeg_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    args = parse_args()
    out = run_inference(args)
    print(f"Done. Output: {out}")
