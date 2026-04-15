"""
This is my inference dataloader.
It loads real-world data and runs inference.

Current main goals:
- Test across different videos.
- Experiment with various time embedding patterns.
- Test with different camera types.
- Perform multi-turn inference to for AR Generation.
"""

import os
import json
import torch
import torch.utils.data
from einops import rearrange
from torchvision.transforms import v2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import random
from .utils import (
    get_relative_pose, get_relative_pose_first_fixed, process_camera_trajectory, compute_pose_embedding,
    compute_pose_embedding_normalized, load_video_with_random_start, crop_and_resize, load_frames_using_imageio
)

from ..utils.builder import DATASETS

@DATASETS.register_module(name='camxtime_evaluation_set')
class CamXTimeEvalDataset(torch.utils.data.Dataset):
    """Dataset class for camxtime evaluation"""
    
    VALID_ASSETS = [
        "src_video",
        "src_camera",
        "tgt_camera",
        "src_video_timestamp",
        "tgt_video_timestamp",
        "text",
        "dataset_name",
        "scene_name",
        "src_camera_idx",
        "tgt_camera_idx",
    ]

    def __init__(self, config=None):
        """Initialize for inference mode - loads raw videos"""

        self.cfg = config
        self.base_path = getattr(self.cfg.dataset, 'data_path', 'camxtime_eval')
        metadata_path = f"{self.base_path}/metadata.csv"
        self.start_from_last_frame = self.cfg.inference.start_from_last
        self.normalize_pose = getattr(self.cfg.inference, 'normalize_pose', False)  # Default to True for backward compatibility
        metadata = pd.read_csv(metadata_path)
        # self.normalize_pose = True
        # Create expanded dataset: each video × each camera
        self.text = []
        self.path = []
        self.cam_type = []
        self.video_idx = []
        self.file_name = []

        # Read available cameras from the extrinsics JSON
        camera_extrinsics_path = os.path.join(self.base_path, "cameras", self.cfg.inference.camera_file)
        with open(camera_extrinsics_path, 'r') as f:
            cam_json = json.load(f)
        if cam_json:
            first_frame_key = list(cam_json.keys())[0]
            all_cameras = sorted([int(k.replace('cam', '')) for k in cam_json[first_frame_key].keys()])
        else:
            all_cameras = [0]  # No external camera extrinsics (e.g. identity eval); use placeholder

        for video_idx in range(len(metadata)):
            for cam_type in all_cameras:
                self.text.append(metadata["text"].iloc[video_idx])
                self.path.append(os.path.join(self.base_path, "videos", metadata["file_name"].iloc[video_idx]))
                self.cam_type.append(cam_type)
                self.video_idx.append(video_idx)
                self.file_name.append(os.path.splitext(metadata["file_name"].iloc[video_idx])[0])

        self.src_camera_path = os.path.join(self.base_path, "src_cam")

        # Video processing parameters
        self.max_num_frames = 81
        self.frame_interval = 1
        self.num_frames = 81
        self.height = 480
        self.width = 832
        self.is_i2v = False

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(self.height, self.width)),
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
                        
    def load_video(self, file_path):
        return load_frames_using_imageio(file_path, self.max_num_frames, 0, 
                                         self.frame_interval, self.num_frames, self.frame_process,
                                         self.width, self.height)

    def cam_data_to_tensor(self, cam_data):
        """Convert camera data to tensor"""
        cam_data_tensor = []
        for cam_idx in range(len(cam_data)):
            cam_data_tensor.append(torch.tensor(cam_data[f"camera_{cam_idx:03d}"]))
        cam_data_tensor = torch.stack(cam_data_tensor, dim=0)
        return cam_data_tensor

    def process_camera_trajectory_blender(self, cam_data_tensor, cam_idx_traj):
        """Process camera trajectory for given frames and camera index"""
        
        w2c = cam_data_tensor[cam_idx_traj]  # Shape: [81, 4, 4]
        
        camera_traj_c2w = []
        for i in range(w2c.shape[0]):
            camera_traj_c2w.append(np.linalg.inv(w2c[i]))
        camera_traj_c2w = np.stack(camera_traj_c2w)

        def c2w_blender_to_opencv(c2w_blender):
            blender_to_opencv = np.eye(4)
            blender_to_opencv[:3, :3] = np.array([
                [1,  0,  0],   
                [ 0, -1,  0],   # flip Y
                [ 0,  0, -1]    # flip Z
            ])        
            c2w_blender_opencv = c2w_blender @ blender_to_opencv
            x_flip = np.eye(4)
            x_flip[0, 0] = -1
            c2w_blender_opencv = x_flip @ c2w_blender_opencv
            return c2w_blender_opencv

        camera_traj_c2w_opencv = []
        for i in range(camera_traj_c2w.shape[0]):
            camera_traj_c2w_opencv.append(c2w_blender_to_opencv(camera_traj_c2w[i]))
        camera_traj_c2w_opencv = np.stack(camera_traj_c2w_opencv)
        
        camera_traj_c2w_opencv = camera_traj_c2w_opencv[::4]  # Downsample to 21 frames
        return camera_traj_c2w_opencv

    def _get_time_pattern(self, pattern: str, T=81):
        """
        Return a predefined temporal frame index pattern for 81-frame sequences.

        Args:
            pattern (str): Name of the temporal pattern.
                Supported:
                    - "normal"                  → [0, 1, 2, ..., 80]
                    - "reverse"                 → [80, 79, ..., 0]
                    - "bounce_40"               → forward 40→80, then back 80→40
                    - "zigzag_15_21_5"          → forward 60→80, then back 80→20
                    - "zigzag_5_21_15"          → forward 20→80, then back 80→60
                    - "repeat_0to40_double"     → 0, 1, 1, 2, 2, ..., 40, 40
                    - "start40_repeat_next"     → 40, 41, 41, 42, 42, ...
                    - "go_and_freeze"           → 0, 1, 2, ..., 40, then 40, 40, 40, ... (freeze at 40)
                    - "fixed_0" / "fixed_5" / "fixed_10" / "fixed_20" → constant time
        Returns:
            List[int | float]: List of 81 frame indices defining the pattern.
        """

        if pattern == "reverse":
            base = list(range(T - 1, -1, -1))

        elif pattern == "bounce_40":
            start = 40
            base = list(range(start, T)) + list(range(T - 1, start - 1, -1))

        elif pattern == "zigzag_15_21_5":
            fa, fb, fc = 4 * 15, 4 * 21, 4 * 5
            base = list(range(fa, fb + 1)) + list(range(fb, fc - 1, -1))

        elif pattern == "zigzag_5_21_15":
            fa, fb, fc = 4 * 5, 4 * 21, 4 * 15
            base = list(range(fa, fb + 1)) + list(range(fb, fc - 1, -1))

        elif pattern == "repeat_0to40_double":
            base = [0] + [i for i in range(1, 41) for _ in (0, 1)]

        elif pattern == "start40_repeat_next":
            base = [40] + [i for i in range(41, T) for _ in (0, 1)]

        elif pattern == "go_and_freeze":
            # 0, 1, 2, ..., 40, then freeze at 40 for remaining frames
            freeze_point = 40
            base = list(range(freeze_point + 1)) + [freeze_point] * (T - freeze_point - 1)

        elif pattern == "normal":
            base = list(range(T))

        elif pattern == "fixed_10":
            base = [40.0] * T

        elif pattern == "fixed_5":
            base = [20.0] * T

        elif pattern == "fixed_0":
            base = [0.0] * T

        elif pattern == "fixed_15":
            base = [60.0] * T

        elif pattern == "fixed_20":
            base = [80.0] * T

        elif pattern == "zigzag_0_10_0":
            # 0 → frame 40 (step 10) → back to 0
            base = list(range(0, 41)) + list(range(39, -1, -1))

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        # --- Ensure exactly 81 elements ---
        if len(base) >= T:
            return base[:T]

        out = []
        i = 0
        while len(out) < T:
            out.append(base[i % len(base)])
            i += 1

        return out

    def _indices_to_time_embedding_21(self, cam_indices_81):
        # Map 81-frame indices to 21-step time indices (0..20), aligned with stride 4
        return [max(0, min(20, idx // 4)) for idx in cam_indices_81]


    def load_src_camera(self, video_path):
        """Load and normalize source camera from JSON file"""
        # Extract video name without extension
        video_name = os.path.basename(video_path).replace('.mp4', '')
        
        # Build path to camera JSON
        camera_dir = os.path.join(self.src_camera_path, video_name)
        camera_json_path = os.path.join(camera_dir, "camera_data.json")
        
        # Load camera data from JSON
        with open(camera_json_path, 'r') as f:
            cam_data = json.load(f)
        
        # Convert to tensor and process trajectory
        cam_data_tensor = self.cam_data_to_tensor(cam_data["cameras"]["extrinsic"])
        cam_idx = list(range(81))  # All 81 frames
        camera_traj_c2w_opencv = self.process_camera_trajectory_blender(cam_data_tensor, cam_idx)
        
        # Normalize relative to first frame
        camera_traj_c2w_opencv_first_frame = camera_traj_c2w_opencv[0]
        camera_traj_c2w_opencv_first_frame_inv = np.linalg.inv(camera_traj_c2w_opencv_first_frame)
        src_c2w_norm = camera_traj_c2w_opencv @ camera_traj_c2w_opencv_first_frame_inv
        
        # Optional: Scene scale normalization
        if self.normalize_pose:
            scale_range = (1.0, 1.1)
            translations = src_c2w_norm[:, :3, 3]
            scene_scale = np.max(np.abs(translations))
            
            if scene_scale < 1e-2:
                scene_scale = 1.0
            else:
                scene_scale *= random.uniform(*scale_range)
                src_c2w_norm[:, :3, 3] /= scene_scale
        
        # Extract 3x4 pose matrices (21 frames)
        final_poses = []
        for i in range(len(src_c2w_norm)):
            final_poses.append(torch.as_tensor(src_c2w_norm[i])[:3, :])
        src_cam = torch.stack(final_poses, dim=0)  # Shape: [21, 3, 4]
        
        return src_cam

    def __getitem__(self, data_id):

        """Get validation/test item - loads raw videos"""
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        
        assert video.shape[1] == 81
        
        # Load source camera from JSON
        data = {}
        src_cam = self.load_src_camera(path)  # path is the video path
        src_cam = rearrange(src_cam, 'b c d -> b (c d)')  # Shape: [21, 12]
        data['src_camera'] = src_cam.to(torch.bfloat16)
        data['text'] = text
        data["src_video"] = video
        data["src_video_timestamp"] = torch.tensor(self._get_time_pattern("normal", 81), dtype=torch.float32)

        camera_extrinsics_path = os.path.join(self.base_path, "cameras", self.cfg.inference.camera_file)
        with open(camera_extrinsics_path, 'r') as file:
            cam_data = json.load(file)
        cam_idx = list(range(video.shape[1]))[::4]
        tgt_cam_params = process_camera_trajectory(cam_data, cam_idx, int(self.cam_type[data_id]))

        # Apply start_from_last_frame logic if enabled
        if self.start_from_last_frame:
            relative_poses = []
            for i in range(len(tgt_cam_params)):
                relative_pose = get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
                relative_poses.append(relative_pose[1])
            starting_pose = relative_poses[-1]
            final_poses = []
            for i in range(len(tgt_cam_params)):
                relative_pose = get_relative_pose_first_fixed([tgt_cam_params[0], tgt_cam_params[i]], starting_pose)
                final_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
            pose_embedding = torch.stack(final_poses, dim=0)  # 21x3x4
            
            # Apply normalization only if enabled
            if self.normalize_pose:
                scale_range = (1.0, 1.1)
                scene_scale = torch.max(torch.abs(pose_embedding[:, :3, 3]))  # Get max translation
                translation_threshold = 1e-2  # Define threshold for meaningful translation
                if scene_scale > translation_threshold:
                    scene_scale = random.uniform(scale_range[0], scale_range[1]) * scene_scale
                    pose_embedding[:, :3, 3] /= scene_scale  # Normalize translations
                else:
                    print(f"Static camera detected (max translation: {scene_scale:.6f}), keeping original scale")
            
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        else:
            # Use normalized or non-normalized version based on config
            if self.normalize_pose:
                pose_embedding = compute_pose_embedding_normalized(tgt_cam_params)
            else:
                pose_embedding = compute_pose_embedding(tgt_cam_params)
                pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')

        # assume all src videos are the identity pose # todo: change this
        data['tgt_camera'] = pose_embedding.to(torch.bfloat16)
        pattern = getattr(self.cfg.inference, 'time_mode', 'normal')
        allowed = {"reverse","bounce_40","zigzag_15_21_5","zigzag_5_21_15","repeat_0to40_double","start40_repeat_next","go_and_freeze","fixed_10","fixed_5","fixed_0","fixed_20", "normal"}
        if pattern in allowed:
            idx_81 = self._get_time_pattern(pattern, video.shape[1])
        else:
            print(f"{pattern} is not a valid time mode")
            print(f"Using normal time mode")
            idx_81 = self._get_time_pattern("normal", video.shape[1])

        data['tgt_video_timestamp'] = torch.tensor(idx_81, dtype=torch.float32)
        # Add video and camera indices for proper multi-GPU handling
        data['video_idx'] = self.video_idx[data_id]
        data['cam_type'] = self.cam_type[data_id]
        data['file_name'] = self.file_name[data_id]

        return data

    def __len__(self):
        return len(self.path)


@DATASETS.register_module(name='camxtime_identity_set')
class CamXTimeIdentityEvalDataset(CamXTimeEvalDataset):
    """Evaluation dataset where target camera = source camera (reconstruction/identity test)."""

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")

        assert video.shape[1] == 81

        src_cam = self.load_src_camera(path)        # [21, 3, 4]
        src_cam = rearrange(src_cam, 'b c d -> b (c d)')  # [21, 12]
        src_cam_bf16 = src_cam.to(torch.bfloat16)

        pattern = getattr(self.cfg.inference, 'time_mode', 'normal')
        allowed = {
            "normal", "reverse",
            "bounce_40", "zigzag_15_21_5", "zigzag_5_21_15", "zigzag_0_10_0",
            "repeat_0to40_double", "start40_repeat_next", "go_and_freeze",
            "fixed_0", "fixed_5", "fixed_10", "fixed_15", "fixed_20",
        }
        if pattern not in allowed:
            print(f"{pattern} is not a valid time mode, using normal")
            pattern = "normal"
        idx_81 = self._get_time_pattern(pattern, video.shape[1])

        return {
            'src_video': video,
            'src_camera': src_cam_bf16,
            'tgt_camera': src_cam_bf16,          # same as source
            'src_video_timestamp': torch.tensor(self._get_time_pattern("normal", 81), dtype=torch.float32),
            'tgt_video_timestamp': torch.tensor(idx_81, dtype=torch.float32),
            'text': text,
            'video_idx': self.video_idx[data_id],
            'cam_type': self.cam_type[data_id],
            'file_name': self.file_name[data_id],
        }

@DATASETS.register_module(name='camxtime_cross_set')
class CamXTimeCrossEvalDataset(CamXTimeEvalDataset):
    """Cross-camera eval: source and target use different camera trajectories per video.

    Expects the dataset directory to contain:
      src_cam/<video_name>/camera_data.json  - source camera trajectory
      tgt_cam/<video_name>/camera_data.json  - target camera trajectory
    Both in the same camera_data.json format as CamXTimeEvalDataset.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.tgt_camera_path = os.path.join(self.base_path, "tgt_cam")

    def load_tgt_camera(self, video_path):
        """Load and normalize target camera from tgt_cam directory."""
        video_name = os.path.basename(video_path).replace('.mp4', '')
        camera_json_path = os.path.join(self.tgt_camera_path, video_name, "camera_data.json")

        with open(camera_json_path, 'r') as f:
            cam_data = json.load(f)

        cam_data_tensor = self.cam_data_to_tensor(cam_data["cameras"]["extrinsic"])
        cam_idx = list(range(81))
        camera_traj_c2w_opencv = self.process_camera_trajectory_blender(cam_data_tensor, cam_idx)

        first_frame_inv = np.linalg.inv(camera_traj_c2w_opencv[0])
        tgt_c2w_norm = first_frame_inv @ camera_traj_c2w_opencv

        if self.normalize_pose:
            scale_range = (1.0, 1.1)
            translations = tgt_c2w_norm[:, :3, 3]
            scene_scale = np.max(np.abs(translations))
            if scene_scale >= 1e-2:
                scene_scale *= random.uniform(*scale_range)
                tgt_c2w_norm[:, :3, 3] /= scene_scale

        final_poses = []
        for i in range(len(tgt_c2w_norm)):
            final_poses.append(torch.as_tensor(tgt_c2w_norm[i])[:3, :])
        return torch.stack(final_poses, dim=0)  # [21, 3, 4]

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        assert video.shape[1] == 81

        src_cam = self.load_src_camera(path)
        src_cam = rearrange(src_cam, 'b c d -> b (c d)')

        tgt_cam = self.load_tgt_camera(path)
        tgt_cam = rearrange(tgt_cam, 'b c d -> b (c d)')

        pattern = getattr(self.cfg.inference, 'time_mode', 'normal')
        allowed = {
            "normal", "reverse",
            "bounce_40", "zigzag_15_21_5", "zigzag_5_21_15", "zigzag_0_10_0",
            "repeat_0to40_double", "start40_repeat_next", "go_and_freeze",
            "fixed_0", "fixed_5", "fixed_10", "fixed_15", "fixed_20",
        }
        if pattern not in allowed:
            print(f"{pattern} is not a valid time mode, using normal")
            pattern = "normal"
        idx_81 = self._get_time_pattern(pattern, video.shape[1])

        return {
            'src_video': video,
            'src_camera': src_cam.to(torch.bfloat16),
            'tgt_camera': tgt_cam.to(torch.bfloat16),
            'src_video_timestamp': torch.tensor(self._get_time_pattern("normal", 81), dtype=torch.float32),
            'tgt_video_timestamp': torch.tensor(idx_81, dtype=torch.float32),
            'text': text,
            'video_idx': self.video_idx[data_id],
            'cam_type': self.cam_type[data_id],
            'file_name': self.file_name[data_id],
        }
