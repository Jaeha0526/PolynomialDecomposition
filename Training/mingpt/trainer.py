"""
Minimal training loop for the polynomial-decomposition model.

Single-GPU only by design: BGRPO and the paper's architecture sweeps all
run on one H100 per job, so there's no need for DataParallel here. Runs
a cosine LR schedule with linear warmup, periodically evaluates on the
validation set, and saves the checkpoint with the lowest validation loss
(suffix ``_best.pt``).
"""

import logging
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


def _wandb_log(payload: dict, step: int) -> None:
    """Log to wandb if a run is active, otherwise no-op. Lets the trainer
    stay neutral about whether the caller wired wandb up."""
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(payload, step=step)


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    # LR schedule: linear warmup, then cosine decay to 10% of the initial LR.
    lr_decay = False
    shuffle = False
    warmup_tokens = 375e6       # defaults are from GPT-3; callers usually override.
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0
    validation_interval = 50    # run validation every N training iterations
    writer = None               # optional tensorboard.SummaryWriter

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """Teacher-forced cross-entropy training with periodic best-checkpointing."""

    def __init__(self, model, train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.optimizer = None

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        self.tokens = 0              # cumulative unpadded tokens (for LR schedule)
        self.valid_loss_best = float("inf")
        self.best_iter = 0

    def save_checkpoint(self, suffix: str = "") -> None:
        """Save current weights alongside ``ckpt_path`` with ``suffix`` inserted before the extension."""
        if self.config.ckpt_path is None:
            return
        path = Path(self.config.ckpt_path)
        if suffix:
            # Split on extension so `_best.pt` style suffixes behave as expected.
            target = path.with_name(path.stem + suffix)
        else:
            target = path
        logger.info("saving %s", target)
        torch.save(self.model.state_dict(), str(target))

    def _build_optimizer(self) -> None:
        no_decay = ("bias", "LayerNorm.weight")
        decay, no_decay_params = [], []
        for n, p in self.model.named_parameters():
            (no_decay_params if any(nd in n for nd in no_decay) else decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = optim.AdamW(
            groups, lr=self.config.learning_rate, betas=self.config.betas
        )

    def _update_lr(self, y: torch.Tensor) -> float:
        config = self.config
        if not config.lr_decay:
            return config.learning_rate

        self.tokens += (y >= 0).sum().item()
        if self.tokens < config.warmup_tokens:
            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
        else:
            progress = float(self.tokens - config.warmup_tokens) / float(
                max(1, config.final_tokens - config.warmup_tokens)
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        lr = config.learning_rate * lr_mult
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _validate(self, loader_valid: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x_valid, y_valid in loader_valid:
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.to(self.device)
                _, loss_valid = self.model(x_valid, y_valid)
                losses.append(loss_valid.mean().item())
        self.model.train()
        return losses[-1] if losses else float("inf")

    def train(self) -> None:
        self._build_optimizer()
        config = self.config
        step = 0
        last_valid_loss: float | None = None

        loader_valid = DataLoader(self.valid_dataset, batch_size=128)

        for epoch in range(config.max_epochs):
            loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=config.shuffle,
            )
            self.model.train()
            pbar = tqdm(enumerate(loader), total=len(loader))

            for it, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                _, loss = self.model(x, y)
                loss = loss.mean()

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                if it % config.validation_interval == 0:
                    last_valid_loss = self._validate(loader_valid)
                    is_best = last_valid_loss < self.valid_loss_best
                    if is_best:
                        self.save_checkpoint("_best.pt")
                        self.best_iter = step
                        self.valid_loss_best = last_valid_loss
                    _wandb_log(
                        {
                            "valid/loss": last_valid_loss,
                            "valid/loss_best": self.valid_loss_best,
                            "valid/best_iter": self.best_iter,
                        },
                        step,
                    )

                lr = self._update_lr(y)

                valid_str = f"{last_valid_loss:.11f}" if last_valid_loss is not None else "pending"
                pbar.set_description(
                    f"epoch {epoch + 1} iter {it}: train loss {loss.item():.11f}"
                    f" valid_loss {valid_str}."
                    f" best saved at iteration {self.best_iter} lr {lr:e}"
                )

                _wandb_log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "epoch": epoch + 1,
                    },
                    step,
                )
                if config.writer is not None:
                    config.writer.add_scalar("train/loss", loss.item(), step)
                    config.writer.add_scalar("train/lr", lr, step)

                step += 1
