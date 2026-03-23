"""
Miscellaneous utility functions for SpaceTimePilot training and validation
"""
import os
import time
import shutil
import imageio
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    """Save video frames to file with progress bar"""
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def dict_to_namespace(d):
    """Recursively convert nested dictionaries to SimpleNamespace"""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d
