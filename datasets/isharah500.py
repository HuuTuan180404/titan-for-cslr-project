"""Dataset adapter for the iSharah500 benchmark."""

from pathlib import Path
from typing import Any, Dict
import warnings

import cv2
import pickle
import numpy as np
import pandas as pd

from .base_dataset import BaseSignLanguageDataset

"""
Pose Data Format
================

Pose data is stored in pickle (.pkl) files under:
    data/isharah500/pose/
        00_0001.pkl
        00_0002.pkl
        ...

Each PKL file contains a dictionary with a ``keypoints`` entry: pose_dict["keypoints"]

The ``keypoints`` array has shape: (T, J, 2)

where:
    - T: Number of frames.
    - J: Number of keypoints.
    - 2: (x, y) coordinates of each keypoint.

Example:

    keypoints = pose_dict["keypoints"]

    right_hand = keypoints[:, 0:21, :] -> Right hand (21 landmarks)
    left_hand  = keypoints[:, 21:42, :] -> Left hand (21 landmarks)
    lips       = keypoints[:, 42:61, :] -> Lips (19 landmarks)
    body       = keypoints[:, 61:, :] -> Body (remaining keypoints)
"""

"""
RGB Data Format
===============

RGB frame data is stored under:
    data/isharah500/rgb/
        00_0001/
            000001.jpg
            000002.jpg
            ...
        00_0002/
            000001.jpg
            000002.jpg
            ...

Each sample directory (e.g. ``00_0001``) contains the RGB frames
of a single video/sample.

Each frame is stored as an image file in JPG, JPEG, or PNG format.

The corresponding cropped hand RGB data is stored under:
    data/isharah500/rgb_cropped/
        00_0001/
            left/
                000001.jpg
                000002.jpg
                ...
            right/
                000001.jpg
                000002.jpg
                ...

The cropped left-hand and right-hand images are resized to: (112, 112, 3)
    where:
        - 112: Image height.
        - 112: Image width.
        - 3: RGB color channels.
"""


class ISharah500Dataset(BaseSignLanguageDataset):
    """iSharah500 dataset built on :class:`BaseSignLanguageDataset`."""

    def __init__(self, config: Any, phase: str, vocab=None):
        super().__init__(config=config, phase=phase, vocab=vocab)

    def _load_pose(self, path: Path) -> Dict[str, np.ndarray]:
        _key = "keypoints"

        with path.open("rb") as file:
            pose_data = pickle.load(file)

        pose_data = pose_data[_key]

        pose = {
            "right": pose_data[:, 0:21, :],
            "left": pose_data[:, 21:42, :],
            "face": pose_data[:, 42:61, :],
            "body": pose_data[:, 61:, :],
        }

        return pose

    def _load_rgb(self, path: Path) -> Any:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"RGB sample directory does not exist: {path}. "
                "If you do not want to use RGB data, set use_rgb=False in the config."
            )

        if self.config.rgb.left_hand and not (path / "left").exists():
            raise FileNotFoundError(
                f"Left hand RGB directory does not exist: {path / 'left'}. "
                "If you do not want to use RGB data, set use_rgb=False in the config."
            )

        if self.config.rgb.right_hand and not (path / "right").exists():
            raise FileNotFoundError(
                f"Right hand RGB directory does not exist: {path / 'right'}. "
                "If you do not want to use RGB data, set use_rgb=False in the config."
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        rgb = {}

        for hand in ("left", "right"):
            hand_path = path / hand
            if not hand_path.is_dir():
                rgb[f"rgb_{hand}"] = []
                continue

            frames = []
            frame_paths = sorted(
                frame_path
                for frame_path in hand_path.iterdir()
                if frame_path.is_file()
                and frame_path.suffix.lower() in image_extensions
            )

            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Unable to read RGB frame: {frame_path}")
                frames.append(frame)

            rgb[f"rgb_{hand}"] = self.resize_images(frames)

        return rgb
