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
    compute_pose_embedding_normalized, load_frames_using_imageio
)

from ..utils.builder import DATASETS

@DATASETS.register_module(name='evaluation_set_movedcam')
class InferenceDataset(torch.utils.data.Dataset):
    """Dataset class for validation/inference only"""
    
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

        self.base_path = config.inference.base_path
        metadata_path = f"{self.base_path}/metadata.csv"
        self.cfg = config
        self.start_from_last_frame = self.cfg.inference.start_from_last
        self.normalize_pose = getattr(self.cfg.inference, 'normalize_pose', False)  # Default to True for backward compatibility
        metadata = pd.read_csv(metadata_path)

        # Create expanded dataset: each video × each camera
        self.text = []
        self.path = []
        self.cam_type = []
        self.video_idx = []
        
        # all_cameras = list(range(1, 11))  # Cameras 1-10, adjust as needed
        all_cameras = [1]

        for video_idx in range(len(metadata)):
            for cam_type in all_cameras:
                self.text.append(metadata["text"].iloc[video_idx])
                self.path.append(os.path.join(self.base_path, "videos", metadata["file_name"].iloc[video_idx]))
                self.cam_type.append(cam_type)
                self.video_idx.append(video_idx)
        
        print("this will process the following number of videos: ", len(self.path))
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

    def _get_time_pattern(self, pattern: str, T=81):
        """
        Return a predefined temporal frame index pattern for 81-frame sequences.

        Args:
            pattern (str): Name of the temporal pattern.
                Supported:
                    - "forward"                 → [0, 1, 2, ..., 80]
                    - "reverse"                 → [80, 79, ..., 0]
                    - "pingpong"                → forward 40→80, then back 80→40
                    - "bounce_late"             → forward 60→80, then back 80→20
                    - "bounce_early"            → forward 20→80, then back 80→60
                    - "slowmo_first_half"       → 0, 1, 1, 2, 2, ..., 40, 40
                    - "slowmo_second_half"      → 40, 41, 41, 42, 42, ...
                    - "ramp_then_freeze"        → 0, 1, 2, ..., 40, then 40, 40, 40, ... (freeze at 40)
                    - "freeze_start" / "freeze_early" / "freeze_mid" / "freeze_late" / "freeze_end" → bullet-time (freeze at frame 0/20/40/60/80)
        Returns:
            List[int | float]: List of 81 frame indices defining the pattern.
        """

        if pattern == "reverse":
            base = list(range(T - 1, -1, -1))

        elif pattern == "pingpong":
            start = 40
            base = list(range(start, T)) + list(range(T - 1, start - 1, -1))

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
            # 0, 1, 2, ..., 40, then freeze at 40 for remaining frames
            freeze_point = 40
            base = list(range(freeze_point + 1)) + [freeze_point] * (T - freeze_point - 1)

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

    def load_src_camera(self, camera_matrices):
        # Convert w2c to c2w by inverting
        src_w2c = camera_matrices
        src_c2w = np.linalg.inv(src_w2c)        # (N, 4, 4)

        # --- Normalize all poses relative to the first source frame ---
        ref_pose = src_c2w[0]                    # reference camera (world aligned to first src)
        ref_pose_inv = np.linalg.inv(ref_pose)
        src_c2w_norm = src_c2w @ ref_pose_inv   # (N, 4, 4)
        
        # Get the maximum absolute translation value
        translations = src_c2w_norm[:, :3, 3]    # Extract translation vectors (N, 3)
        scene_scale = np.max(np.abs(translations))
        
        if scene_scale < 1e-2:
            scene_scale = 1.0

        scene_scale = 1
        src_c2w_norm[:, :3, 3] /= scene_scale
        
        src_c2w_norm = src_c2w_norm[::4]
        final_poses = []
        for i in range(len(src_c2w_norm)):
            final_poses.append(torch.as_tensor(src_c2w_norm[i])[:3,:]) 
        src_cam = torch.stack(final_poses, dim=0)  # 21x3x4
        return src_cam

    def __getitem__(self, data_id):

        """Get validation/test item - loads raw videos"""
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        
        assert video.shape[1] == 81
        
        # # load src camera
        src_camera_path = os.path.join(self.src_camera_path, f"{path.split('/')[-1].split('.')[0]}_extrinsics.npy")
        raw_src_cam = np.load(src_camera_path)
        src_cam = self.load_src_camera(raw_src_cam)
        src_cam = rearrange(src_cam, 'b c d -> b (c d)')

        data = {}
        data['src_camera'] = src_cam.to(torch.bfloat16)
        data['text'] = text
        data["src_video"] = video
        data["src_video_timestamp"] = torch.tensor(self._get_time_pattern("forward", 81), dtype=torch.float32)

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
                scene_scale = torch.max(torch.abs(pose_embedding[:, :3, 3]))  # Get max translation
                translation_threshold = 1e-2  # Define threshold for meaningful translation
                if scene_scale > translation_threshold:
                    scene_scale = 1.0 * scene_scale
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

        pattern = getattr(self.cfg.inference, 'time_mode', 'forward')
        allowed = {"forward","reverse","pingpong","bounce_late","bounce_early","slowmo_first_half","slowmo_second_half","ramp_then_freeze","freeze_start","freeze_early","freeze_mid","freeze_late","freeze_end"}
        if pattern in allowed:
            idx_81 = self._get_time_pattern(pattern, video.shape[1])
        else:
            print(f"{pattern} is not a valid time mode")
            print(f"Using forward time mode")
            idx_81 = self._get_time_pattern("forward", video.shape[1])

        data['tgt_video_timestamp'] = torch.tensor(idx_81, dtype=torch.float32)
        # Add video and camera indices for proper multi-GPU handling
        data['video_idx'] = self.video_idx[data_id]
        data['cam_type'] = self.cam_type[data_id]
        
        return data

    def __len__(self):
        return len(self.path)