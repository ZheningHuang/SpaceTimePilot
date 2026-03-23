import os
import torch
import numpy as np
import imageio
import torchvision
from PIL import Image
from einops import rearrange
import random


def resample_to_fixed_length(seq, target_len=81):
    """Resample a sequence to exactly target_len using linear interpolation."""
    # get the minimal start frame
    max_start = len(seq) - target_len
    
    # If sequence is shorter than target, we can't crop, so we need to repeat/extend
    if max_start < 0:
        # For shorter sequences, just repeat the sequence to reach target length
        while len(seq) < target_len:
            seq = seq + seq
        max_start = len(seq) - target_len
    
    start = random.randint(0, max_start) if max_start > 0 else 0
    seq = seq[start:]
    idxs = list(range(0, target_len))
    return [seq[i] for i in idxs]

def time_augmentation(num_frames=81, mode="forward", pivot=None, seg_start=None, seg_len=10, fixed_frame=None):
    frames = list(range(num_frames))

    # 1. Forward
    if mode == "forward":
        raw = frames

    # 2. Backward
    elif mode == "backward":
        raw = frames[::-1]

    # 3. Boomerang (forward + full reverse)
    elif mode == "boomerang":
        raw = frames + frames[::-1]

    # 4. Zigzag (forward to pivot then reverse back)
    elif mode == "zigzag":
        if pivot is None:
            pivot = random.randint(num_frames//2 + 1, num_frames - 1)  # random pivot
        forward = list(range(pivot+1))
        backward = list(range(pivot-1, -1, -1))
        raw = forward + backward

    # 5. Slow global (duplicate every frame)
    elif mode == "slow_global":
        raw = [f for f in frames for _ in (0,1)]  # double length

    # 6. Segment slow motion (duplicate inside chosen segment)
    elif mode == "slow_segment":
        # Use at least 20 frames for slow motion segment
        min_seg_len = 20
        max_seg_len = 25
        seg_len = random.randint(min_seg_len, max_seg_len)
        actual_seg_len = seg_len    
        if seg_start is None:
            # random segment start
            seg_start = random.randint(0, num_frames - actual_seg_len)
        seg_end = min(seg_start + actual_seg_len, num_frames)
        raw = []
        for i in range(num_frames):
            raw.append(i)
            if seg_start <= i < seg_end:
                raw.append(i)  # duplicate in slow region

    # 7. Random fixed frame (all indices point to same random frame - static video)
    elif mode == "fixed_frame":
        if fixed_frame is None:
            fixed_frame = random.randint(0, num_frames - 1)  # random frame
        raw = [fixed_frame] * num_frames  # all frames are the same

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return resample_to_fixed_length(raw)


def random_time_augmentation(num_frames=81, augmentation_probs=None):
    if augmentation_probs is None:
        augmentation_probs = {
            "forward": 0.45,     # 45%
            "backward": 0.1,     # 10% 
            "boomerang": 0.1,    # 10%
            "zigzag": 0.1,       # 10%
            "slow_global": 0.1,  # 10%
            "slow_segment": 0.1, # 10%
            "fixed_frame": 0.05  # 5%
        }
    
    # Validate probabilities sum to 1.0
    total_prob = sum(augmentation_probs.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
    
    # Random selection based on probabilities
    modes = list(augmentation_probs.keys())
    probabilities = list(augmentation_probs.values())
    
    # Use random.choices for weighted selection
    selected_mode = random.choices(modes, weights=probabilities)[0]
    
    # Apply the selected augmentation
    sequence = time_augmentation(num_frames=num_frames, mode=selected_mode)
    
    return sequence, selected_mode


class Camera:
    """Camera class for handling camera parameters and transformations"""
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    """Parse matrix string into numpy array"""
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def get_relative_pose(cam_params):
    """Calculate relative poses between cameras"""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    
    # Use identity matrix for consistency across modes
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def get_relative_pose_first_fixed(cam_params, target_cam_c2w):
    """Get relative pose with fixed first camera pose"""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def process_camera_trajectory(cam_data, frame_indices, cam_idx):
    """Process camera trajectory for given frames and camera index"""
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in frame_indices]
    traj = np.stack(traj)
    traj = traj.transpose(0, 2, 1) # parsing the read data from row vector to column vector
    c2ws = []
    for c2w in traj:
        # Apply coordinate system transformations
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.
        c2w[:3, 3] /= 100
        c2ws.append(c2w)
    return [Camera(cam_param) for cam_param in c2ws]

def process_camera_trajectory_blender(cam_data, frame_indices, cam_idx):
    """Process camera trajectory for given frames and camera index"""
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in frame_indices]
    traj = np.stack(traj)
    traj = traj.transpose(0, 2, 1) # parsing the read data from row vector to column vector
    c2ws = []
    for c2w in traj:
        # Apply coordinate system transformations
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.
        c2w[:3, 3] /= 100
        c2ws.append(c2w)
    return [Camera(cam_param) for cam_param in c2ws]

def compute_pose_embedding(cam_params_list):
    """Compute pose embedding from camera parameters"""
    relative_poses = []
    for i in range(len(cam_params_list)):
        relative_pose = get_relative_pose([cam_params_list[0], cam_params_list[i]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)
    return pose_embedding.to(torch.bfloat16)

def compute_pose_embedding_normalized(cam_params_list):
    """Compute pose embedding from camera parameters and normalize the translation"""
    relative_poses = []
    for i in range(len(cam_params_list)):
        relative_pose = get_relative_pose([cam_params_list[0], cam_params_list[i]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)
    scale_range = (1.0, 1.1)
    scene_scale = torch.max(torch.abs(pose_embedding[:, :3, 3]))  # Get max translation
    translation_threshold = 1e-2  # Define threshold for meaningful translation
    if scene_scale > translation_threshold:
        scene_scale = random.uniform(scale_range[0], scale_range[1]) * scene_scale
        pose_embedding[:, :3, 3] /= scene_scale  # Normalize translations
        print(f"Normalized translation: {scene_scale:.6f}")
    else:
        print(f"Static camera detected (max translation: {scene_scale:.6f}), keeping original scale")
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
    return pose_embedding.to(torch.bfloat16)


def crop_and_resize(image, target_width, target_height):
    """Crop and resize image to target dimensions"""
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image, (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    return image


def load_frames_using_imageio(file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process, target_width, target_height, is_i2v=False):
    """Load video frames using imageio"""
    reader = imageio.get_reader(file_path)
    total_frames = reader.count_frames()
    
    # Adjust max_num_frames and num_frames to actual video length
    actual_max_frames = min(max_num_frames, total_frames)
    required_frames = start_frame_id + (num_frames - 1) * interval + 1
    
    if total_frames < required_frames:
        # If video is too short, adjust parameters to use what we have
        actual_num_frames = min(num_frames, (total_frames - start_frame_id + interval - 1) // interval)
        if actual_num_frames <= 0:
            reader.close()
            return None
    else:
        actual_num_frames = num_frames
    
    frames = []
    first_frame = None
    for frame_id in range(actual_num_frames):
        frame_idx = min(start_frame_id + frame_id * interval, total_frames - 1)
        frame = reader.get_data(frame_idx)
        frame = Image.fromarray(frame)
        frame = crop_and_resize(frame, target_width, target_height)
        if first_frame is None:
            first_frame = np.array(frame)
        frame = frame_process(frame)
        frames.append(frame)
    
    # If we have fewer frames than expected, pad by repeating the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])  # Repeat last frame
    
    reader.close()

    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames if not is_i2v else (frames, first_frame)


def load_video_with_random_start(file_path, max_num_frames, frame_interval, num_frames, frame_process, target_width, target_height, is_i2v=False):
    """Load video with random start frame"""
    # start_frame_id = torch.randint(0, max_num_frames - (num_frames - 1) * frame_interval, (1,))[0]
    start_frame_id = 0
    return load_frames_using_imageio(file_path, max_num_frames, start_frame_id, frame_interval, num_frames, frame_process, target_width, target_height, is_i2v)
