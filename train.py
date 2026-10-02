import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from datasets.utils import cslr_collate_fn
from datasets.isharah500 import ISharah500Dataset

from models.model import CSLRModel

from losses.total_loss import TotalLoss

from utils.logger import get_logger
from utils.plot import plot_training_history
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import ctc_greedy_decode, decode_targets, compute_wer


def load_config(config_path: str) -> DictConfig:
    """Load YAML config and resolve variable interpolation."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)

    # Resolve ${...} references
    OmegaConf.resolve(config)

    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    device = config.training.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    return torch.device(device)


def build_dataset(config, phase: str, vocab=None):
    dataset_name = config.dataset.benchmark.lower()

    if dataset_name == "isharah500":
        pass

    return ISharah500Dataset(config=config, phase=phase, vocab=vocab)


def build_dataloader(dataset, config, phase: str, collate_fn=None):
    is_train = phase == "train"

    return DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle if is_train else False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=(
            config.dataloader.persistent_workers and config.dataloader.num_workers > 0
        ),
        collate_fn=collate_fn,
    )


def build_model(cfg) -> nn.Module:
    model_name = cfg.model.name.lower()
    if model_name == "mymodel":
        model = CSLRModel(cfg=cfg)
        return model
    raise ValueError(f"Unknown model: {model_name}")


def build_loss(config):
    if config.loss.name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(
            label_smoothing=config.loss.get("label_smoothing", 0.0)
        )

    raise ValueError(f"Unknown loss: {config.loss.name}")


def build_optimizer(model, cfg):
    optimizer_config = cfg.optimizer

    name = optimizer_config.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("eps", 1e-8),
        )

    raise ValueError(f"Unknown optimizer: {optimizer_config.name}")


def build_scheduler(optimizer, cfg):
    scheduler_name = cfg.scheduler.name.lower()

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.scheduler.eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, device, blank_id):
    model.train()

    total_loss = 0.0
    total_samples = 0

    total_errors = 0
    total_words = 0

    for batch in dataloader:
        # ==================================================
        # Input
        # ==================================================

        lh_pose = batch["left"].to(device)
        rh_pose = batch["right"].to(device)

        lh_rgb = batch["rgb_left"].to(device) if batch["rgb_left"] is not None else None

        rh_rgb = (
            batch["rgb_right"].to(device) if batch["rgb_right"] is not None else None
        )

        face = batch["face"].to(device) if batch["face"] is not None else None

        body = batch["body"].to(device) if batch["body"] is not None else None

        gloss_ids = batch["gloss_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)

        # ==================================================
        # Forward
        # ==================================================
        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        # ==================================================
        # CTC targets
        # ==================================================
        target_ids = torch.cat(batch["target_ids"]).to(device)

        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # ==================================================
        # Loss
        # ==================================================
        loss_dict = criterion(outputs, target_ids, input_lengths, target_lengths)

        loss = loss_dict["loss"]

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ==================================================
        # Loss statistics
        # ==================================================

        batch_size = lh_pose.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        # ==================================================
        # CTC decoding
        # ==================================================

        ctc_logits = outputs["ctc_logits"]

        predictions = ctc_greedy_decode(
            logits=ctc_logits, input_lengths=input_lengths, blank_id=blank_id
        )

        targets = decode_targets(target_ids=target_ids, target_lengths=target_lengths)

        # ==================================================
        # WER
        # ==================================================
        batch_errors, batch_words = compute_wer(predictions, targets)
        total_errors += batch_errors
        total_words += batch_words

    # ======================================================
    # Epoch statistics
    # ======================================================
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    wer = total_errors / total_words if total_words > 0 else 0.0

    return avg_loss, wer


@torch.no_grad()
def validate(model, dataloader, criterion, device, blank_id):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_errors = 0
    total_words = 0

    for batch in dataloader:
        lh_pose = batch["left"].to(device)
        rh_pose = batch["right"].to(device)

        lh_rgb = batch["rgb_left"].to(device) if batch["rgb_left"] is not None else None

        rh_rgb = (
            batch["rgb_right"].to(device) if batch["rgb_right"] is not None else None
        )

        face = batch["face"].to(device) if batch["face"] is not None else None

        body = batch["body"].to(device) if batch["body"] is not None else None

        gloss_ids = batch["gloss_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)

        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        target_ids = torch.cat(batch["target_ids"]).to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        loss_dict = criterion(outputs, target_ids, input_lengths, target_lengths)

        loss = loss_dict["loss"]

        batch_size = lh_pose.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        ctc_logits = outputs["ctc_logits"]

        # [B, T, C]
        predictions = ctc_greedy_decode(
            logits=ctc_logits, input_lengths=input_lengths, blank_id=blank_id
        )

        targets = decode_targets(target_ids=target_ids, target_lengths=target_lengths)

        batch_errors, batch_words = compute_wer(predictions, targets)
        total_errors += batch_errors
        total_words += batch_words

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    wer = total_errors / total_words if total_words > 0 else 0.0

    return avg_loss, wer


@torch.no_grad()
def inference(model, dataloader, device, blank_id):
    model.eval()

    total_errors = 0
    total_words = 0

    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Inference", unit="batch"):
        lh_pose = batch["left"].to(device)
        rh_pose = batch["right"].to(device)

        lh_rgb = batch["rgb_left"].to(device) if batch["rgb_left"] is not None else None

        rh_rgb = (
            batch["rgb_right"].to(device) if batch["rgb_right"] is not None else None
        )

        face = batch["face"].to(device) if batch["face"] is not None else None

        body = batch["body"].to(device) if batch["body"] is not None else None

        gloss_ids = batch["gloss_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)

        outputs = model(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, gloss_ids, text_attention_mask
        )

        ctc_logits = outputs["ctc_logits"]

        input_lengths = batch["input_lengths"].to(device)

        predictions = ctc_greedy_decode(
            logits=ctc_logits, input_lengths=input_lengths, blank_id=blank_id
        )

        target_ids = torch.cat(batch["target_ids"]).to(device)

        target_lengths = batch["target_lengths"].to(device)

        targets = decode_targets(target_ids=target_ids, target_lengths=target_lengths)

        batch_errors, batch_words = compute_wer(predictions, targets)

        total_errors += batch_errors
        total_words += batch_words

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    wer = total_errors / total_words if total_words > 0 else 0.0

    return wer, all_predictions, all_targets


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    max_frames = cfg.dataset.max_frames

    seed = cfg.project.seed

    save_ckpt_dir = Path(cfg.checkpoint.save_dir)
    save_ckpt_dir = save_ckpt_dir / cfg.dataset.benchmark
    save_ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = cfg.training.epochs

    best_dev_wer = float("inf")

    history = {
        "train_loss": [],
        "train_wer": [],
        "dev_loss": [],
        "dev_wer": [],
    }

    set_seed(seed)

    device = get_device(cfg)

    logger = get_logger(
        name="cslr", log_dir=f"{cfg.logging.log_dir}/{cfg.dataset.benchmark}"
    )

    train_set = build_dataset(cfg, phase="train")
    vocab = train_set.vocab
    dev_set = build_dataset(cfg, phase="dev", vocab=vocab)

    train_loader = build_dataloader(
        train_set,
        cfg,
        "train",
        lambda batch: cslr_collate_fn(batch, vocab=vocab, max_frames=max_frames),
    )
    dev_loader = build_dataloader(
        dev_set,
        cfg,
        "dev",
        lambda batch: cslr_collate_fn(batch, vocab=vocab, max_frames=max_frames),
    )

    model = build_model(cfg=cfg)
    model = model.to(device)

    criterion = TotalLoss(cfg)

    optimizer = build_optimizer(model, cfg)

    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    if cfg.training.resume:
        start_epoch, best_dev_wer = load_checkpoint(
            path=save_ckpt_dir / cfg.checkpoint.resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(
            f"Resume training from checkpoint: \n"
            f"\t\tStart epoch      : {start_epoch}\n"
            f"\t\tBest Dev WER     : {best_dev_wer * 100:.2f}%"
        )
    else:
        logger.info("Start training")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_wer = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_set.blank_id
        )

        dev_loss, dev_wer = validate(
            model, dev_loader, criterion, device, train_set.blank_id
        )

        history["train_loss"].append(train_loss)
        history["train_wer"].append(train_wer)
        history["dev_loss"].append(dev_loss)
        history["dev_wer"].append(dev_wer)

        save_checkpoint(
            path=save_ckpt_dir / "last_model.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_dev_wer=best_dev_wer,
            train_loss=train_loss,
            train_wer=train_wer,
            dev_loss=dev_loss,
            dev_wer=dev_wer,
            vocab=vocab,
        )

        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            save_path = save_ckpt_dir / "best_model.pth"
            save_checkpoint(
                path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_dev_wer=best_dev_wer,
                train_loss=train_loss,
                train_wer=train_wer,
                dev_loss=dev_loss,
                dev_wer=dev_wer,
                vocab=vocab,
            )
            logger.info(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train WER: {train_wer * 100:.2f}% "
                f"| Val Loss: {dev_loss:.4f} "
                f"| Val WER: {dev_wer * 100:.2f}%"
                f"| → Best model saved: {save_path}"
            )
        else:
            logger.info(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train WER: {train_wer * 100:.2f}% "
                f"| Val Loss: {dev_loss:.4f} "
                f"| Val WER: {dev_wer * 100:.2f}%"
            )

        scheduler.step()

    if cfg.plot.enabled:
        plot_training_history(
            history, save_dir=f"{cfg.logging.log_dir}/{cfg.dataset.benchmark}"
        )

    logger.info("TESTING THE BEST CHECKPOINT")
    logger.info("START INFERENCE")

    best_ckpt_path = save_ckpt_dir / "best_model.pth"

    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")

    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(best_ckpt["model_state_dict"])

    ckp_vocab = best_ckpt["vocab"]

    test_set = build_dataset(cfg, phase="test", vocab=ckp_vocab)
    test_loader = build_dataloader(
        test_set,
        cfg,
        "test",
        lambda batch: cslr_collate_fn(batch, vocab=ckp_vocab, max_frames=max_frames),
    )

    wer, _, _ = inference(model, test_loader, device, test_set.blank_id)

    logger.info(f"Test WER: {wer * 100:.2f}%")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/isharah500.yaml")

    args = parser.parse_args()

    main(args.config)
