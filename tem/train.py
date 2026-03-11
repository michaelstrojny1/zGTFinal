from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_rmse: float
    history: list[dict]
    total_seconds: float


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def evaluate(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    use_amp = amp and device.type == "cuda"
    se_sum = 0.0
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x).squeeze(-1).float()
            se_sum += torch.sum((pred - y) ** 2).item()
            n += y.numel()
    return math.sqrt(se_sum / max(1, n))


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    amp: bool = True,
    grad_clip: float = 1.0,
    compile_model: bool = True,
    low_rul_loss_weight: float = 1.0,
    low_rul_threshold: float = 30.0,
    low_rul_weight_power: float = 1.0,
) -> TrainResult:
    model = model.to(device)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)  # type: ignore[assignment]
        except Exception as err:  # pragma: no cover - backend dependent
            warnings.warn(f"torch.compile unavailable, continuing without compile: {err}")

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=use_fused)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=5.0, reduction="none")
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    start = time.perf_counter()
    use_amp = amp and device.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        num_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x).squeeze(-1)
                per_sample_loss = criterion(pred, y)
                if low_rul_loss_weight > 1.0 and low_rul_threshold > 0.0:
                    # Emphasize low-RUL samples to improve terminal-life point accuracy.
                    rel = torch.clamp((low_rul_threshold - y) / low_rul_threshold, min=0.0, max=1.0)
                    if low_rul_weight_power != 1.0:
                        rel = torch.pow(rel, low_rul_weight_power)
                    weights = 1.0 + (low_rul_loss_weight - 1.0) * rel
                    loss = torch.mean(per_sample_loss * weights)
                else:
                    loss = torch.mean(per_sample_loss)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            num_batches += 1

        scheduler.step()
        val_rmse = evaluate(model, val_loader, device=device, amp=amp)
        avg_train_loss = loss_sum / max(1, num_batches)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_rmse": val_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

    total_seconds = time.perf_counter() - start
    return TrainResult(
        best_state_dict=best_state,
        best_val_rmse=best_val_rmse,
        history=history,
        total_seconds=total_seconds,
    )
