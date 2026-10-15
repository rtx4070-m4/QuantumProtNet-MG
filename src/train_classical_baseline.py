import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
EMBED_DIM = 24
CHANNELS = 32
DROPOUT = 0.15

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "hf_mirror"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_ID = 0
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AA_TO_ID) + 1


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(split):
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seq = row["aa_seq"].strip().upper()
            unknown = set(seq) - set(AMINO_ACIDS)
            if unknown:
                raise ValueError(
                    f"Unexpected amino-acid symbols {unknown} in {row['name']}"
                )
            if len(seq) != int(row["protein_length"]):
                raise ValueError(f"Length mismatch in {row['name']}")

            rows.append({
                "name": row["name"],
                "parent": row["parent"],
                "sequence": seq,
                "label": float(row["label"]),
            })

    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


class ProteinDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = [AA_TO_ID[aa] for aa in row["sequence"]]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(row["label"], dtype=torch.float32),
        )


def collate_batch(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)

    tokens = torch.full(
        (len(sequences), max_len), PAD_ID, dtype=torch.long
    )
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = seq

    mask = tokens != PAD_ID
    labels = torch.stack(labels)
    return tokens, mask, labels


# ---------------------------------------------------------
# Custom neural network built from primitive operations
# ---------------------------------------------------------
class MultiScaleGatedRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID
        )

        self.local_branches = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, CHANNELS, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])

        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.gate = nn.Conv1d(EMBED_DIM, CHANNELS, kernel_size=1)
        self.norm = nn.LayerNorm(CHANNELS)

        self.regressor = nn.Sequential(
            nn.Linear(CHANNELS * 2, 48),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, tokens, mask):
        # tokens: [batch, length]
        # mask:   [batch, length]
        embedded = self.embedding(tokens).transpose(1, 2)

        # Learned weighting of three local sequence scales.
        branch_features = torch.stack(
            [torch.nn.functional.gelu(branch(embedded))
             for branch in self.local_branches],
            dim=1,
        )
        weights = torch.softmax(self.scale_logits, dim=0)
        combined = (
            branch_features * weights.view(1, 3, 1, 1)
        ).sum(dim=1)

        # Adaptive feature gate.
        gate = torch.sigmoid(self.gate(embedded))
        features = (combined * gate).transpose(1, 2)
        features = self.norm(features)

        mask3 = mask.unsqueeze(-1)
        masked_features = features * mask3.to(features.dtype)

        # Padding is excluded from both pooling operations.
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = masked_features.sum(dim=1) / lengths.to(features.dtype)

        max_features = features.masked_fill(
            ~mask3, torch.finfo(features.dtype).min
        )
        max_pool = max_features.max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.regressor(pooled).squeeze(1)


def calculate_metrics(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    error = y_pred - y_true

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denominator = torch.sqrt(
        (true_centered ** 2).sum() * (pred_centered ** 2).sum()
    )
    pearson = (
        (true_centered * pred_centered).sum() / denominator
    ).item() if denominator.item() > 0 else None

    return {"MAE": mae, "RMSE": rmse, "Pearson_r": pearson}


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    targets = []
    predictions = []

    for tokens, mask, labels in loader:
        tokens = tokens.to(DEVICE)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(tokens, mask)
        losses.append(loss_fn(preds, labels).item())
        targets.append(labels.cpu())
        predictions.append(preds.cpu())

    y = torch.cat(targets)
    p = torch.cat(predictions)
    metrics = calculate_metrics(y, p)
    metrics["MSE_loss"] = sum(losses) / len(losses)
    return metrics


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    train_rows = read_csv("train")
    valid_rows = read_csv("valid")
    test_rows = read_csv("test")

    print("Train rows:", len(train_rows))
    print("Validation rows:", len(valid_rows))
    print("Test rows:", len(test_rows))

    # Record the known official-split overlap for transparent reporting.
    train_sequences = {r["sequence"] for r in train_rows}
    test_sequences = {r["sequence"] for r in test_rows}
    train_parents = {r["parent"] for r in train_rows}
    test_parents = {r["parent"] for r in test_rows}

    print("Exact train-test sequence overlap:",
          len(train_sequences & test_sequences))
    print("Train-test shared parent identifiers:",
          len(train_parents & test_parents))

    train_loader = DataLoader(
        ProteinDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        ProteinDataset(valid_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ProteinDataset(test_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = MultiScaleGatedRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    loss_fn = nn.MSELoss()

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable parameters:", parameter_count)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = RESULT_DIR / "classical_baseline_best.pt"

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        seen = 0

        for tokens, mask, labels in train_loader:
            tokens = tokens.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(tokens, mask)
            loss = loss_fn(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            seen += batch_n

        train_mse = total_loss / seen
        val_metrics = evaluate(model, valid_loader, loss_fn)

        record = {
            "epoch": epoch,
            "train_MSE": train_mse,
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_Pearson_r": val_metrics["Pearson_r"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train MSE {train_mse:.4f} | "
            f"val MAE {val_metrics['MAE']:.4f} | "
            f"val RMSE {val_metrics['RMSE']:.4f} | "
            f"val r {val_metrics['Pearson_r']} | "
            f"{time.time() - start:.1f}s"
        )

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "validation_MAE": best_val_mae,
                "config": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "embedding_dim": EMBED_DIM,
                    "channels": CHANNELS,
                    "dropout": DROPOUT,
                    "learning_rate": LEARNING_RATE,
                    "amino_acids": AMINO_ACIDS,
                },
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    # Restore the checkpoint selected using validation MAE.
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test is evaluated only after model selection is complete.
    test_metrics = evaluate(model, test_loader, loss_fn)

    print("\nBest epoch:", best_epoch)
    print("Best validation MAE:", round(best_val_mae, 5))
    print("Official test metrics:", test_metrics)

    if DEVICE.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print("Peak allocated GPU memory (MiB):", round(peak_memory_mb, 2))
    else:
        peak_memory_mb = None

    with (RESULT_DIR / "classical_baseline_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    results = {
        "model": "Custom multiscale gated sequence regressor",
        "device": str(DEVICE),
        "pytorch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_MAE": best_val_mae,
        "official_test_metrics": test_metrics,
        "train_test_exact_sequence_overlap": len(
            train_sequences & test_sequences
        ),
        "train_test_shared_parent_ids": len(
            train_parents & test_parents
        ),
        "peak_gpu_memory_MiB": peak_memory_mb,
        "seed": SEED,
        "note": (
            "Official test results are subject to the documented "
            "train-test sequence and parent overlap."
        ),
    }

    with (RESULT_DIR / "classical_baseline_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nSaved checkpoint:", checkpoint_path)
    print("Saved history:", RESULT_DIR / "classical_baseline_history.csv")
    print("Saved metrics:", RESULT_DIR / "classical_baseline_metrics.json")


if __name__ == "__main__":
    main()
