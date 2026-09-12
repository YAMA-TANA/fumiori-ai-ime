"""Residual LoRA Fine-Tuning for Ruri-v3-70m IME Reranker.

Tunes a lightweight rank=16 LoRA on ModernBert Wqkv, Wo, Wi layers
using the targeted residual disambiguation dataset (curated + public-corpus pairs).
Directly fixes failure cases while preserving established Mozc & holdout accuracy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Transformers 4.57 eagerly imports its optional scikit-learn generation
# helpers when AutoModel is imported.  On the Windows build used for this
# project that optional native stack can terminate the interpreter before
# model loading (pandas/scipy access violation).  LoRA training does not use
# those helpers, so make sklearn unavailable only for this process.
_find_spec = importlib.util.find_spec


def _find_spec_without_sklearn(name: str, *args: Any, **kwargs: Any):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


class IMERerankerDataset(Dataset):
    def __init__(self, data_path: Path):
        self.samples = json.loads(data_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, max_length: int = 128) -> Dict[str, torch.Tensor]:
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch]

    pos_enc = tokenizer(
        queries,
        positives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    neg_enc = tokenizer(
        queries,
        negatives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "pos_input_ids": pos_enc["input_ids"],
        "pos_attention_mask": pos_enc["attention_mask"],
        "neg_input_ids": neg_enc["input_ids"],
        "neg_attention_mask": neg_enc["attention_mask"],
    }


def train_residual_lora(
    base_model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 32,
    grad_accum_steps: int = 2,
    lr: float = 1.5e-4,
    margin: float = 1.2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    print(f"Loading Tokenizer and 70M Student Model from {base_model_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)

    model = ModernBertForSequenceClassification.from_pretrained(
        str(base_model_dir),
        num_labels=1,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    print(f"Applying PEFT LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["Wqkv", "Wo", "Wi"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training dataset from {dataset_path}...", flush=True)
    dataset = IMERerankerDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=(device == "cuda"),
    )

    total_steps = (len(loader) // grad_accum_steps) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )
    ranking_criterion = nn.MarginRankingLoss(margin=margin)

    print(f"Training for {epochs} epochs | Total batches: {len(loader)} | Effective steps: {total_steps}...", flush=True)
    step = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for idx, batch in enumerate(loader, start=1):
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            pos_logits = model(pos_ids, attention_mask=pos_mask).logits.squeeze(-1)
            neg_logits = model(neg_ids, attention_mask=neg_mask).logits.squeeze(-1)

            # 1. Main Margin Ranking Loss: force pos_logits >= neg_logits + margin
            rank_loss = ranking_criterion(pos_logits, neg_logits, target=torch.ones_like(pos_logits))

            # 2. Calibration regularizer: encourage pos to be positive and neg to be negative
            calib_loss = 0.1 * (torch.relu(-pos_logits).mean() + torch.relu(neg_logits).mean())

            loss = (rank_loss + calib_loss) / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if idx % grad_accum_steps == 0 or idx == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 200 == 0 or step == total_steps:
                    elapsed = time.perf_counter() - t0
                    cur_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / idx
                    pos_mean = pos_logits.mean().item()
                    neg_mean = neg_logits.mean().item()
                    print(
                        f"Epoch {epoch}/{epochs} | Step {step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | "
                        f"Pos: {pos_mean:+.2f} | Neg: {neg_mean:+.2f} | "
                        f"Gap: {pos_mean - neg_mean:+.2f} | Elapsed: {elapsed:.1f}s",
                        flush=True,
                    )

    # Merge LoRA back into base model
    print("Training finished! Merging LoRA weights back into the base ModernBert model...", flush=True)
    merged_model = model.merge_and_unload()

    # Backup existing model directory before overwriting
    backup_dir = base_model_dir.parent / f"{base_model_dir.name}-bak"
    if not backup_dir.exists():
        print(f"Creating backup of original model at {backup_dir}...", flush=True)
        shutil.copytree(str(base_model_dir), str(backup_dir))

    print(f"Saving merged updated 70M model to {output_dir}...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[SUCCESS] Updated 70M model successfully saved to {output_dir}!", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=ROOT / "models" / "ruri-v3-70m-ime-distilled")
    parser.add_argument("--dataset", type=Path, default=ROOT / "integration" / "ime_residual_lora_train.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ruri-v3-70m-ime-distilled")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    train_residual_lora(
        base_model_dir=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
