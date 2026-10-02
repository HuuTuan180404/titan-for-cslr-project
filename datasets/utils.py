import torch
from torch.nn.utils.rnn import pad_sequence


def pad_or_truncate_frames(x: torch.Tensor, max_frames, padding_value=0.0):
    """
    Đưa sequence về đúng max_frames.

    Args:
        x: Tensor [T, ...]
        max_frames: số frame mong muốn

    Returns:
        x: Tensor [max_frames, ...]
        length: số frame thật
    """

    T = x.shape[0]

    # T > max_frames: truncate
    if T > max_frames:
        # raise ValueError(
        #     f"Number of frames ({T}) exceeds the maximum allowed ({max_frames})."
        # )
        indices = torch.linspace(
            0,
            T - 1,
            steps=max_frames,
            device=x.device,
        ).long()

        x = x[indices]

        length = max_frames
    # T < max_frames: padding
    elif T < max_frames:
        pad_shape = (max_frames - T, *x.shape[1:])

        padding = torch.full(
            pad_shape, fill_value=padding_value, dtype=x.dtype, device=x.device
        )

        x = torch.cat([x, padding], dim=0)
        length = T

    # T == max_frames
    else:
        length = max_frames

    return x, length


def create_frame_attention_mask(lengths: torch.Tensor, max_frames: int):
    """
    Tạo mask cho frame.

    1 = frame thật
    0 = frame padding

    Args:
        lengths: [B]
        max_frames: int

    Returns:
        [B, max_frames]
    """

    positions = torch.arange(max_frames, device=lengths.device).unsqueeze(0)

    mask = positions < lengths.unsqueeze(1)

    return mask.long()


def encode_gloss(gloss, vocab):
    """
    Gloss tokens -> token IDs.
    """

    unk_idx = vocab["<UNK>"]

    token_ids = [vocab.get(token, unk_idx) for token in gloss]

    return torch.tensor(token_ids, dtype=torch.long)


def create_text_attention_mask(gloss_ids: torch.Tensor, pad_idx: int):
    """
    1 = gloss thật
    0 = <pad>
    """

    return (gloss_ids != pad_idx).long()


def cslr_collate_fn(batch, vocab, max_frames=256):
    """
    Collate function cho CSLR.
    """

    # ========================================================
    # Metadata
    # ========================================================

    ids = [sample["id"] for sample in batch]

    texts = [sample["text"] for sample in batch]

    use_rgb = batch[0]["use_rgb"]

    # ========================================================
    # Frame lengths
    # ========================================================

    input_lengths = torch.tensor(
        [min(sample["length"], max_frames) for sample in batch], dtype=torch.long
    )

    # ========================================================
    # Frame attention mask
    # ========================================================

    frame_attention_mask = create_frame_attention_mask(
        lengths=input_lengths, max_frames=max_frames
    )

    # ========================================================
    # Pose
    # ========================================================

    pose_batch = {}

    for key in ["right", "left", "face", "body"]:
        data = []

        for sample in batch:
            x = sample.get(key)

            if x is None:
                data = None
                break

            x, _ = pad_or_truncate_frames(x=x, max_frames=max_frames, padding_value=0.0)

            data.append(x)

        if data is None:
            pose_batch[key] = None
        else:
            pose_batch[key] = torch.stack(data, dim=0)

    # ========================================================
    # RGB
    # ========================================================

    rgb_batch = {}

    if use_rgb:
        for key in ["rgb_left", "rgb_right"]:
            data = []

            for sample in batch:
                x = sample.get(key)

                if x is None:
                    data = None
                    break

                x, _ = pad_or_truncate_frames(
                    x=x, max_frames=max_frames, padding_value=0.0
                )

                data.append(x)

            if data is None:
                rgb_batch[key] = None
            else:
                rgb_batch[key] = torch.stack(data, dim=0)

    else:
        rgb_batch["rgb_left"] = None
        rgb_batch["rgb_right"] = None

    # ========================================================
    # Gloss -> IDs
    # ========================================================

    gloss_ids = []

    for sample in batch:
        ids_ = encode_gloss(gloss=sample["gloss"], vocab=vocab)

        gloss_ids.append(ids_)

    # ========================================================
    # Gloss lengths
    # ========================================================

    target_lengths = torch.tensor([len(x) for x in gloss_ids], dtype=torch.long)
    target_ids = gloss_ids.copy()

    # ========================================================
    # Padding gloss
    # ========================================================

    pad_idx = vocab["<PAD>"]

    gloss_ids = pad_sequence(gloss_ids, batch_first=True, padding_value=pad_idx)

    # ========================================================
    # Text attention mask
    # ========================================================

    text_attention_mask = create_text_attention_mask(
        gloss_ids=gloss_ids, pad_idx=pad_idx
    )

    # ========================================================
    # Return
    # ========================================================

    return {
        "id": ids,
        "text": texts,
        "use_rgb": use_rgb,
        # Pose
        "right": pose_batch["right"],
        "left": pose_batch["left"],
        "face": pose_batch["face"],
        "body": pose_batch["body"],
        # RGB
        "rgb_left": rgb_batch["rgb_left"],
        "rgb_right": rgb_batch["rgb_right"],
        # Visual
        "input_lengths": input_lengths,
        "frame_attention_mask": frame_attention_mask,
        # Text
        "gloss_ids": gloss_ids,
        "target_lengths": target_lengths,
        "text_attention_mask": text_attention_mask,
        "target_ids": target_ids,
    }
