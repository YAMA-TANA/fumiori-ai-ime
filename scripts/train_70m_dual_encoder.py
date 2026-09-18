"""Train 70M ModernBERT Dual-Encoder (Bi-Encoder) for ultra-low latency IME reranking."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Avoid sklearn import issue on Windows
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    ModernBertModel,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


class ModernBertDualEncoder(nn.Module):
    """Siamese ModernBert Dual-Encoder with Mean Pooling and L2 Normalization."""

    def __init__(self, base_model: ModernBertModel) -> None:
        super().__init__()
        self.encoder = base_model

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBert outputs last_hidden_state
        last_hidden = outputs.last_hidden_state
        pooled = self.mean_pooling(last_hidden, attention_mask)
        # L2 normalize so dot product equals cosine similarity
        return F.normalize(pooled, p=2, dim=1)


class ContrastiveDualEncoderDataset(Dataset):
    def __init__(self, data_path: Path):
        self.samples = json.loads(data_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, max_len: int = 64) -> Dict[str, torch.Tensor]:
    contexts = [b["context"] for b in batch]
    positives = [b["positive"] for b in batch]
    # Pick first hard negative if available, else fallback to reading
    negatives = [b["negatives"][0] if b.get("negatives") else b["reading"] for b in batch]

    ctx_enc = tokenizer(contexts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    pos_enc = tokenizer(positives, padding=True, truncation=True, max_length=16, return_tensors="pt")
    neg_enc = tokenizer(negatives, padding=True, truncation=True, max_length=16, return_tensors="pt")

    return {
        "ctx_ids": ctx_enc["input_ids"],
        "ctx_mask": ctx_enc["attention_mask"],
        "pos_ids": pos_enc["input_ids"],
        "pos_mask": pos_enc["attention_mask"],
        "neg_ids": neg_enc["input_ids"],
        "neg_mask": neg_enc["attention_mask"],
    }


def info_nce_loss(
    ctx_embeds: torch.Tensor,
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss with in-batch negatives plus hard negatives."""
    batch_size = ctx_embeds.size(0)
    
    # 1. Similarity with positive candidates (B x B)
    # Diagonal contains true (ctx_i, pos_i) pairs, off-diagonal acts as in-batch negatives
    sim_pos = torch.matmul(ctx_embeds, pos_embeds.T) / temperature
    
    # 2. Similarity with hard negative candidates (B x B or B x 1)
    sim_neg = torch.matmul(ctx_embeds, neg_embeds.T) / temperature
    
    # Concatenate all candidate similarities for each context: [sim_pos, sim_neg] -> (B x 2B)
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    
    # Ground truth targets: the diagonal index 0, 1, 2, ..., B-1 in sim_pos
    targets = torch.arange(batch_size, device=ctx_embeds.device)
    
    return F.cross_entropy(logits, targets)


def train_dual_encoder(
    base_model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    temperature: float = 0.05,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print(f"Loading tokenizer from {base_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)

    print(f"Loading base ModernBertModel from {base_model_dir}...")
    base_model = ModernBertModel.from_pretrained(str(base_model_dir), torch_dtype=torch.float32)

    # Configure LoRA on attention projection layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wqkv", "Wo"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_encoder = get_peft_model(base_model, lora_config)
    peft_encoder.print_trainable_parameters()

    model = ModernBertDualEncoder(peft_encoder).to(device)

    dataset = ContrastiveDualEncoderDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print(f"Starting training: {epochs} epochs, {len(dataset)} samples, {total_steps} total steps...")
    t0 = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_count = 0
        correct_top1 = 0
        total_items = 0

        for step, batch in enumerate(dataloader, start=1):
            ctx_ids = batch["ctx_ids"].to(device)
            ctx_mask = batch["ctx_mask"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            pos_mask = batch["pos_mask"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            neg_mask = batch["neg_mask"].to(device)

            optimizer.zero_grad()

            ctx_embeds = model(ctx_ids, ctx_mask)
            pos_embeds = model(pos_ids, pos_mask)
            neg_embeds = model(neg_ids, neg_mask)

            loss = info_nce_loss(ctx_embeds, pos_embeds, neg_embeds, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

            # Monitor In-Batch Accuracy
            with torch.no_grad():
                # Compare pos score vs neg score for each item
                pos_sim = (ctx_embeds * pos_embeds).sum(dim=1)
                neg_sim = (ctx_embeds * neg_embeds).sum(dim=1)
                correct_top1 += (pos_sim > neg_sim).sum().item()
                total_items += ctx_embeds.size(0)

        avg_loss = epoch_loss / max(1, step_count)
        acc = correct_top1 / max(1, total_items) * 100.0
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | In-Batch (Pos > HardNeg) Acc: {acc:.1f}%")

    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Merge LoRA back into base model for maximum inference performance
    print("Merging LoRA weights into base model...")
    merged_encoder = peft_encoder.merge_and_unload()
    
    # Save PyTorch model and tokenizer
    print(f"Saving merged Dual-Encoder model to {output_dir}...")
    merged_encoder.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "model_type": "modernbert_dual_encoder",
        "embedding_dim": 384,
        "temperature": temperature,
        "epochs": epochs,
        "train_time_s": round(train_time, 2),
        "final_loss": round(avg_loss, 4),
        "final_acc": round(acc, 2),
    }
    (output_dir / "dual_encoder_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Dual-Encoder model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, default=ROOT / "models" / "ruri-v3-70m-base")
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "build" / "dual_encoder_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "dual-encoder-70m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_dual_encoder(
        base_model_dir=args.base_model_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
