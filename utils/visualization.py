import cv2
import numpy as np
import matplotlib.pyplot as plt


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

def visualize_pose_sequence(sample_id, left_hand, right_hand, body=None, face=None, num_frames=6, figsize_per_frame=(4, 4)):
    """
    Visualize evenly sampled frames from a pose sequence.

    Parameters
    ----------
    sample_id : str
        Sample identifier.

    left_hand : np.ndarray
        Shape: (T, 21, 2)

    right_hand : np.ndarray | None
        Shape: (T, 21, 2)

    body : np.ndarray | None
        Shape: (T, N, 2)

    face : np.ndarray | None
        Shape: (T, N, 2)

    num_frames : int
        Number of frames to visualize.

    figsize_per_frame : tuple[int, int]
        Figure size per frame.
    """

    T = left_hand.shape[0]




    # ========================================================
    # Sample frames evenly
    # ========================================================

    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        1, len(frame_indices),
        figsize=(
            figsize_per_frame[0] * len(frame_indices),
            figsize_per_frame[1],
        ),
    )

    # Nếu chỉ visualize 1 frame
    if len(frame_indices) == 1:
        axes = [axes]

    # ========================================================
    # Determine plot range from pose coordinates
    # ========================================================

    x_max = np.nanmax(left_hand[..., 0])
    y_max = np.nanmax(left_hand[..., 1])

    if right_hand is not None:
        x_max = max(x_max, np.nanmax(right_hand[..., 0]))
        y_max = max(y_max, np.nanmax(right_hand[..., 1]))

    if body is not None:
        x_max = max(x_max, np.nanmax(body[..., 0]))
        y_max = max(y_max, np.nanmax(body[..., 1]))

    if face is not None:
        x_max = max(x_max, np.nanmax(face[..., 0]))
        y_max = max(y_max, np.nanmax(face[..., 1]))

    # Thêm một chút margin
    x_max *= 1.05
    y_max *= 1.05

    # ========================================================
    # Visualize
    # ========================================================

    for ax, frame_idx in zip(axes, frame_indices):

        # ----------------------------------------------------
        # Left hand
        # ----------------------------------------------------
        lh = left_hand[frame_idx]
        ax.scatter(lh[:, 0], lh[:, 1], s=9)
        for i, j in HAND_CONNECTIONS:
            ax.plot([lh[i, 0], lh[j, 0]], [lh[i, 1], lh[j, 1]])

        # ----------------------------------------------------
        # Right hand
        # ----------------------------------------------------
        if right_hand is not None:
            rh = right_hand[frame_idx]
            ax.scatter(rh[:, 0], rh[:, 1], s=9)
            for i, j in HAND_CONNECTIONS:
                ax.plot([rh[i, 0], rh[j, 0]], [rh[i, 1], rh[j, 1]])

        # ----------------------------------------------------
        # Body
        # ----------------------------------------------------
        if body is not None:
            b = body[frame_idx]
            ax.scatter(b[:, 0], b[:, 1], s=20)

        # ----------------------------------------------------
        # Face
        # ----------------------------------------------------
        if face is not None:
            f = face[frame_idx]
            ax.scatter(f[:, 0], f[:, 1], s=3)

        # ----------------------------------------------------
        # Plot settings
        # ----------------------------------------------------
        ax.set_title(f"Frame {frame_idx}")

        # Pixel coordinate system:
        # (0, 0) ở góc trên bên trái
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_max, 0)

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

    # ========================================================
    # Figure title
    # ========================================================

    fig.suptitle(f"Sample: {sample_id}", fontsize=14)

    plt.tight_layout()
    plt.show()


def visualize_rgb_sequence(sample_id, what_hand, images: np.ndarray, num_images: int = 6):
    """
    Visualize uniformly sampled images from a sequence.

    Parameters
    ----------
    images : np.ndarray
        Shape: (T, H, W, 3)

    num_images : int
        Number of images to visualize.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected images with shape (T, H, W, 3), "
            f"but got {images.shape}"
        )

    T = images.shape[0]

    # Số frame thực tế cần lấy
    num_images = min(num_images, T)

    # Chọn frame cách đều từ đầu -> cuối
    indices = np.linspace(0, T - 1, num_images, dtype=int)

    # Tính số hàng/cột
    ncols = 4
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    # Ẩn subplot thừa
    for ax in axes[num_images:]:
        ax.axis("off")

    fig.suptitle(f"Sample: {sample_id} - {what_hand}", fontsize=14)

    plt.tight_layout()
    plt.show()
