# datasets/base_dataset.py

import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import cv2
import warnings

import pandas as pd
from torch.utils.data import Dataset


class BaseSignLanguageDataset(Dataset, ABC):
    """
    Base Dataset cho các bài toán Sign Language Recognition.

    Dataset hierarchy:

        Benchmark
            └── Split
                  └── Sample
                        ├── Pose
                        └── RGB

    Ví dụ:

        data/
        ├── WLASL100/
        │   ├── train/
        │   ├── dev/
        │   └── test/
        │
        └── LSA64/
            ├── train/
            ├── dev/
            └── test/
    """

    SUPPORTED_PHASES = {"train", "dev", "test"}

    def __init__(self, config, phase: str, vocab=None):
        """
        Args:
            root:
                Root directory chứa data.

                Ví dụ:
                    ./data

            benchmark:
                Tên benchmark.

                Ví dụ:
                    WLASL100
                    LSA64
                    CSL_Daily

            split:
                train / dev / test

            metadata_root:
                Root directory chứa metadata.

                Nếu None:
                    metadata sẽ nằm trong ./metadata
        """

        super().__init__()
        self.config = config
        self.phase = phase

        self.data_root = Path(config.dataset.root)
        self.benchmark = config.dataset.benchmark
        self.metadata_dir = Path(config.dataset.metadata_root)
        self.use_rgb = config.dataset.use_rgb

        self._validate()

        self.pose_transform = config.pose.transform
        self.pose_normalize = config.pose.normalize
        self.pose_augmentation = config.pose.augmentation
        self.pose_extention = config.pose.format

        self.rgb_transform = config.rgb.transform
        self.rgb_normalize = config.rgb.normalize
        self.rgb_augmentation = config.rgb.augmentation
        self.rgb_extention = config.rgb.format
        self.image_size = config.rgb.image_size

        # Metadata sẽ được load bởi class con
        self.metadata = self._load_metadata()

        self.vocab = vocab
        if vocab is None and self.phase != "test":
            self.vocab = self._build_vocab()
            self.vocab["<PAD>"] = 0  # Padding token
            self.vocab["<UNK>"] = len(self.vocab)
            self.vocab["<BLANK>"] = len(self.vocab)
        if self.vocab is None:
            raise ValueError(
                f"Vocabulary is None for phase='{self.phase}'. "
                "Please provide `vocab` when creating the test dataset."
            )
        self.blank_id = self.vocab["<BLANK>"]
        self.unk_id = self.vocab["<UNK>"]

    def _validate(self):
        """Kiểm tra cấu hình Dataset."""

        if self.phase not in self.SUPPORTED_PHASES:
            raise ValueError(
                f"Unsupported split: '{self.phase}'. "
                f"Expected one of {self.SUPPORTED_PHASES}"
            )

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Benchmark directory does not exist: {self.data_root}"
            )

    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata của benchmark.

        Mặc định:

            metadata/
            └── WLASL100/
                ├── train.csv
                ├── dev.csv
                └── test.csv
        """

        metadata_path = self.metadata_dir / f"{self.phase}.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        metadata = pd.read_csv(metadata_path)

        if len(metadata) == 0:
            raise ValueError(f"Metadata is empty: {metadata_path}")

        if "sample_id" not in metadata.columns:
            raise ValueError(
                f"Metadata must contain 'sample_id' column. "
                f"Found: {list(metadata.columns)}"
            )

        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load một sample.

        Returns:
            Dictionary, ví dụ:

            {
                "id": "sample_000001",
                "gloss": [...],
                "text": "...",
                "use_rgb": True/False,

                'right': <(T, J, C)>,
                'left': <(T, J, C)>,
                'face': <(T, J, C)> | None,
                'body': <(T, J, C)> | None,

                "rgb_left": <(T, H, W, 3)> | None,
                "rgb_right": <(T, H, W, 3)> | None

                "length": 120, ?
            }
        """

        row = self.metadata.iloc[index]

        sample_id = self.get_sample_id(row.name)

        sample: Dict[str, Any] = {
            "id": sample_id,
            "gloss": self.get_optional_value(row, "gloss", default="").split(),
            "text": self.get_optional_value(row, "text", default=""),
            "use_rgb": self.use_rgb,
        }

        # =========================
        # Pose
        # =========================
        pose_path = self.get_pose_path(sample_id)

        if pose_path is not None:
            pose_data = self._load_pose(pose_path)

            for key, value in pose_data.items():
                if value is not None:
                    pose_data[key] = torch.as_tensor(value, dtype=torch.float32)

            sample.update(pose_data)

        # =========================
        # RGB
        # =========================
        if self.use_rgb:
            rgb_path = self.get_rgb_path(sample_id)

            rgb_data = self._load_rgb(rgb_path)

            for key, value in rgb_data.items():
                if value is not None:
                    rgb_data[key] = torch.as_tensor(value, dtype=torch.float32)

                    # (T, H, W, C) -> (T, C, H, W)
                    if key in ["rgb_left", "rgb_right"]:
                        rgb_data[key] = rgb_data[key].permute(0, 3, 1, 2)

            sample.update(rgb_data)

        # =========================
        # Length
        # =========================
        for key in ["right", "left", "face", "body", "rgb_left", "rgb_right"]:
            if sample.get(key) is not None:
                sample["length"] = sample[key].shape[0]
                break
            else:
                sample["length"] = 0

        return sample

    @abstractmethod
    def _load_rgb(self, path: Path) -> Dict[str, np.ndarray]:
        """
        Returns:
            Dictionary, ví dụ:

            {
                "rgb_left": <(T, H, W, 3)>,
                "rgb_right": <(T, H, W, 3)>
            }
        """
        raise NotImplementedError

    @abstractmethod
    def _load_pose(self, path: Path) -> Dict[str, np.ndarray]:
        """
        Returns:
            Dictionary, ví dụ:
            {
                'right': <(T, J, C)>,
                'left': <(T, J, C)>,
                'face': <(T, J, C)>,
                'body': <(T, J, C)>
            }
        """
        raise NotImplementedError

    def get_pose_path(self, sample_id: str) -> Path:
        """
        Trả về path tới pose của sample.

        Ví dụ:

            data/WLASL100/pose/sample_000001.npz
        """

        path = self.data_root / "pose" / f"{sample_id}.{self.pose_extention}"

        if not path.exists():
            warnings.warn(f"Sample '{sample_id}' does not exist: {path}", UserWarning)
            return None
        return path

    def get_rgb_path(self, sample_id: str) -> Path:
        """
        Trả về root directory của RGB sample.

        Ví dụ:

            data/WLASL100/rgb/sample_000001/
        """

        path = self.data_root / "rgb" / sample_id

        return path

    def get_rgb_hand_path(self, sample_id: str, hand: str) -> Path:
        """
        Trả về directory RGB của một bàn tay.

        Args:
            sample_id:
                sample_000001

            hand:
                left / right

        Returns:

            data/WLASL100/rgb/
                sample_000001/
                    left/
                        image1
                        image2
                        ...
                    right/
                        ...

        """

        if hand not in {"left", "right"}:
            raise ValueError(f"Invalid hand: {hand}. Expected 'left' or 'right'.")

        return self.get_rgb_path(sample_id) / hand

    def get_sample_id(self, index: int) -> str:
        """Lấy sample_id theo index."""

        return str(self.metadata.iloc[index]["sample_id"])

    def get_metadata(self, index: int) -> pd.Series:
        """Lấy metadata của một sample."""

        return self.metadata.iloc[index]

    def resize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Resize a sequence of images.

        Parameters
        ----------
        frames : np.ndarray
            Shape: (T, H, W, C)
        size : tuple[int, int]
            Target size: (width, height)

        Returns
        -------
        np.ndarray
            Shape: (T, target_H, target_W, C)
        """
        size = (self.image_size, self.image_size)

        resized_frames = np.stack(
            [cv2.resize(image, size) for image in images],
            axis=0,
        )

        return resized_frames

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"benchmark='{self.benchmark}', "
            f"split='{self.phase}', "
            f"samples={len(self)}"
            f")"
        )

    @staticmethod
    def get_optional_value(row: pd.Series, column: str, default: Any = None) -> Any:
        """Return a metadata value without leaking pandas NaN values."""

        if column not in row.index or pd.isna(row[column]):
            return default
        return row[column]

    def _build_vocab(self):
        glosses = set()
        for gloss in self.metadata["gloss"]:
            if pd.notna(gloss):
                glosses.update(gloss.split())
        return {g: i + 1 for i, g in enumerate(sorted(glosses))}
