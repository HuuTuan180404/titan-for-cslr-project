from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(history, save_dir):
    """
    Plot training history.

    Args:
        history (dict):
            {
                "train_loss": [...],
                "train_wer": [...],
                "dev_loss": [...],
                "dev_wer": [...],
            }
        save_dir (str or Path):
            Directory to save plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)

    plt.plot(epochs, history["dev_loss"], label="Dev Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_dir / "loss_curve.png", dpi=200)

    plt.close()

    # =========================
    # WER
    # =========================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, history["train_wer"], label="Train WER", linewidth=2)

    plt.plot(epochs, history["dev_wer"], label="Dev WER", linewidth=2)

    # Best Dev WER
    best_idx = min(range(len(history["dev_wer"])), key=lambda i: history["dev_wer"][i])

    best_epoch = best_idx + 1
    best_dev_wer = history["dev_wer"][best_idx]

    plt.scatter(
        best_epoch,
        best_dev_wer,
        s=80,
        zorder=5,
        label=f"Best Dev WER ({best_dev_wer:.2f}%)",
    )

    plt.xlabel("Epoch")
    plt.ylabel("WER (%)")
    plt.title("Training and Validation WER")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_dir / "wer_curve.png", dpi=200)

    plt.close()

    print(f"Training curves saved to: {save_dir}")
