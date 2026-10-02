import os
import torch
from pathlib import Path


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_dev_wer,
    train_loss,
    train_wer,
    dev_loss,
    dev_wer,
    vocab,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dev_wer": best_dev_wer,
        "train_loss": train_loss,
        "train_wer": train_wer,
        "dev_loss": dev_loss,
        "dev_wer": dev_wer,
        "vocab": vocab,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_dev_wer = checkpoint["best_dev_wer"]

    print(f"Resume training from epoch {start_epoch}")

    print(f"Best Dev WER: {best_dev_wer * 100:.2f}%")

    return start_epoch, best_dev_wer
