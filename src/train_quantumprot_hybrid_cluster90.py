import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_classical_baseline as baseline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = int(os.environ.get("QP_SEED", "42"))
BATCH_SIZE = baseline.BATCH_SIZE
EPOCHS = baseline.EPOCHS
PATIENCE = baseline.PATIENCE
LEARNING_RATE = baseline.LEARNING_RATE

N_QUBITS = 4
N_Q_LAYERS = 2

RESULT_DIR = Path(os.environ.get("QP_RESULT_DIR", str(ROOT / "results" / "cluster90_hybrid_seed42")))
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = baseline.DEVICE
DIM = 2 ** N_QUBITS


def normalize_parent(value):
    value = str(value or "").strip()
    return re.sub(r"_rd[123]_", "_", value)



def load_controlled_splits():
    manifest_path = Path(os.environ.get(
        "QP_MANIFEST",
        str(ROOT / "results" / "ablation_summary" /
            "scaffold_heldout_development_split_seed20261015.csv")
    ))
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    raw_train = baseline.read_csv("train")
    rows_by_name = {}
    for row in raw_train:
        rows_by_name.setdefault(row["name"], []).append(row)

    train_rows, valid_rows = [], []

    with manifest_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        for item in csv.DictReader(handle):
            name = item["name"]
            candidates = rows_by_name.get(name, [])
            expected_length = int(item["protein_length"])
            expected_label = float(item["label"])
            expected_parent = item["original_parent"].strip()

            match_index = next(
                (
                    i for i, candidate in enumerate(candidates)
                    if len(candidate["sequence"]) == expected_length
                    and abs(candidate["label"] - expected_label) <= 1e-5
                    and str(candidate["parent"]).strip() == expected_parent
                ),
                None,
            )
            if match_index is None:
                raise RuntimeError(
                    f"No matching raw row for manifest record: {name}; "
                    f"length={expected_length}, label={expected_label}, "
                    f"parent={expected_parent}"
                )

            row = candidates.pop(match_index)

            partition = item.get("cluster_partition", item["partition"])
            if partition in ("cluster_development_train", "development_train"):
                train_rows.append(row)
            elif partition in ("cluster_heldout_validation", "group_heldout_validation"):
                valid_rows.append(row)
            else:
                raise RuntimeError(
                    f"Unexpected partition {partition!r} for {name}"
                )

    if not train_rows or not valid_rows:
        raise RuntimeError(
            f"Empty partition: train={len(train_rows)}, "
            f"validation={len(valid_rows)}"
        )

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))

    if not manifest_rows:
        raise RuntimeError("Manifest is empty.")

    if "cluster_id" in manifest_rows[0] and "cluster_partition" in manifest_rows[0]:
        cluster_partitions = {}
        for item in manifest_rows:
            cluster_partitions.setdefault(item["cluster_id"], set()).add(
                item["cluster_partition"]
            )
        crossing = [cid for cid, parts in cluster_partitions.items() if len(parts) > 1]
        if crossing:
            raise RuntimeError(
                f"Cluster leakage detected: {len(crossing)} clusters cross partitions."
            )
        print("\nCLUSTER-AWARE DEVELOPMENT SPLIT")
        print("Manifest:", manifest_path)
        print("Development training rows:", len(train_rows))
        print("Held-out validation rows:", len(valid_rows))
        print("Clusters crossing partitions:", len(crossing))
    else:
        train_groups = {
            item["candidate_group_v4"] for item in manifest_rows
            if item["partition"] == "development_train"
        }
        valid_groups = {
            item["candidate_group_v4"] for item in manifest_rows
            if item["partition"] == "group_heldout_validation"
        }
        if train_groups & valid_groups:
            raise RuntimeError("Candidate-group leakage detected.")
        print("\nSCAFFOLD-HELD-OUT DEVELOPMENT SPLIT")
        print("Manifest:", manifest_path)
        print("Development training rows:", len(train_rows))
        print("Held-out validation rows:", len(valid_rows))
        print("Shared candidate groups:", len(train_groups & valid_groups))

    print("Official test set: NOT LOADED")

    # Existing training loop expects three values. The third is the same
    # validation set for legacy reporting only, not an independent test.
    return train_rows, valid_rows, valid_rows

class QuantumProtHybridRegressor(nn.Module):
    """
    Custom sequence encoder plus a differentiable statevector circuit.

    The simulator uses PyTorch operations only. It does not execute
    on quantum hardware and does not imply quantum advantage.
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            baseline.VOCAB_SIZE,
            baseline.EMBED_DIM,
            padding_idx=baseline.PAD_ID,
        )

        self.local_branches = nn.ModuleList([
            nn.Conv1d(
                baseline.EMBED_DIM,
                baseline.CHANNELS,
                kernel_size=k,
                padding=k // 2,
            )
            for k in (3, 5, 7)
        ])

        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.gate = nn.Conv1d(
            baseline.EMBED_DIM, baseline.CHANNELS, kernel_size=1
        )
        self.norm = nn.LayerNorm(baseline.CHANNELS)

        pooled_dim = baseline.CHANNELS * 2

        # Learned map from sequence features to rotation angles.
        self.angle_projection = nn.Linear(pooled_dim, N_QUBITS)

        # Trainable circuit rotations.
        self.quantum_ry = nn.Parameter(
            0.1 * torch.randn(N_Q_LAYERS, N_QUBITS)
        )
        self.quantum_rz = nn.Parameter(
            0.1 * torch.randn(N_Q_LAYERS, N_QUBITS)
        )

        # Basis indices and measurement signs.
        basis = torch.arange(DIM, dtype=torch.long)

        for q in range(N_QUBITS):
            bit = 1 << (N_QUBITS - 1 - q)
            idx0 = basis[(basis & bit) == 0]
            idx1 = basis[(basis & bit) != 0]

            self.register_buffer(f"idx0_{q}", idx0)
            self.register_buffer(f"idx1_{q}", idx1)
            self.register_buffer(
                f"bitmask_{q}", (basis & bit) != 0
            )
            self.register_buffer(
                f"zsign_{q}",
                torch.where(
                    (basis & bit) == 0,
                    torch.ones(DIM),
                    -torch.ones(DIM),
                ),
            )

        # Each CNOT connects q to the next qubit in a ring.
        for control in range(N_QUBITS):
            target = (control + 1) % N_QUBITS
            control_bit = 1 << (N_QUBITS - 1 - control)
            target_bit = 1 << (N_QUBITS - 1 - target)

            permutation = basis.clone()
            flip = (basis & control_bit) != 0
            permutation[flip] = permutation[flip] ^ target_bit

            self.register_buffer(
                f"cnot_perm_{control}", permutation
            )

        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim + N_QUBITS, 48),
            nn.GELU(),
            nn.Dropout(baseline.DROPOUT),
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def sequence_features(self, tokens, mask):
        embedded = self.embedding(tokens).transpose(1, 2)

        branches = torch.stack([
            torch.nn.functional.gelu(branch(embedded))
            for branch in self.local_branches
        ], dim=1)

        weights = torch.softmax(self.scale_logits, dim=0)
        combined = (
            branches * weights.view(1, 3, 1, 1)
        ).sum(dim=1)

        gate = torch.sigmoid(self.gate(embedded))
        features = self.norm((combined * gate).transpose(1, 2))

        mask3 = mask.unsqueeze(-1)
        masked = features * mask3.to(features.dtype)

        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = masked.sum(dim=1) / lengths.to(features.dtype)

        max_pool = features.masked_fill(
            ~mask3, torch.finfo(features.dtype).min
        ).max(dim=1).values

        return torch.cat([mean_pool, max_pool], dim=1)

    def apply_ry(self, state, theta, qubit):
        # theta has one angle per example.
        c = torch.cos(theta / 2).to(state.real.dtype).unsqueeze(1)
        s = torch.sin(theta / 2).to(state.real.dtype).unsqueeze(1)

        idx0 = getattr(self, f"idx0_{qubit}")
        idx1 = getattr(self, f"idx1_{qubit}")

        a0 = state.index_select(1, idx0)
        a1 = state.index_select(1, idx1)

        new0 = c * a0 - s * a1
        new1 = s * a0 + c * a1

        state = state.index_copy(1, idx0, new0)
        state = state.index_copy(1, idx1, new1)
        return state

    def apply_rz(self, state, theta, qubit):
        # RZ applies opposite complex phases to |0> and |1>.
        # theta: [batch]; state: [batch, 16].
        half = theta / 2

        phase0 = torch.complex(
            torch.cos(half), -torch.sin(half)
        ).unsqueeze(1)
        phase1 = torch.complex(
            torch.cos(half), torch.sin(half)
        ).unsqueeze(1)

        bitmask = getattr(self, f"bitmask_{qubit}").unsqueeze(0)
        phase = torch.where(bitmask, phase1, phase0)
        return state * phase

    def quantum_circuit(self, angles):
        batch_size = angles.size(0)

        state = torch.zeros(
            batch_size, DIM,
            device=angles.device,
            dtype=torch.complex64,
        )
        state[:, 0] = 1.0 + 0.0j

        # Data encoding.
        for q in range(N_QUBITS):
            state = self.apply_ry(state, angles[:, q], q)

        # Trainable variational circuit.
        for layer in range(N_Q_LAYERS):
            for q in range(N_QUBITS):
                theta_y = self.quantum_ry[layer, q].expand(batch_size)
                theta_z = self.quantum_rz[layer, q].expand(batch_size)

                state = self.apply_ry(state, theta_y, q)
                state = self.apply_rz(state, theta_z, q)

            for control in range(N_QUBITS):
                permutation = getattr(self, f"cnot_perm_{control}")
                state = state.index_select(1, permutation)

        # Probabilities and Pauli-Z expectation values.
        probabilities = state.real.square() + state.imag.square()

        expectations = []
        for q in range(N_QUBITS):
            signs = getattr(self, f"zsign_{q}")
            expectations.append((probabilities * signs).sum(dim=1))

        return torch.stack(expectations, dim=1)

    def forward(self, tokens, mask):
        pooled = self.sequence_features(tokens, mask)

        angles = torch.pi * torch.tanh(self.angle_projection(pooled))
        quantum_features = self.quantum_circuit(angles)

        combined = torch.cat([pooled, quantum_features], dim=1)
        return self.regressor(combined).squeeze(1)


def main():
    baseline.set_seed(SEED)

    # Make supported CUDA operations reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_rows, valid_rows, test_rows = load_controlled_splits()

    train_loader = DataLoader(
        baseline.ProteinDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=baseline.collate_batch,
    )
    valid_loader = DataLoader(
        baseline.ProteinDataset(valid_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=baseline.collate_batch,
    )
    test_loader = DataLoader(
        baseline.ProteinDataset(test_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=baseline.collate_batch,
    )

    model = QuantumProtHybridRegressor().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Qubits:", N_QUBITS)
    print("Quantum layers:", N_Q_LAYERS)
    print("Statevector amplitudes:", DIM)
    print("Trainable parameters:", parameter_count)

    # Check forward computation and differentiability before full training.
    model.train()
    tokens, mask, labels = next(iter(train_loader))
    tokens, mask, labels = (
        tokens.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
    )

    optimizer.zero_grad(set_to_none=True)
    predictions = model(tokens, mask)
    smoke_loss = loss_fn(predictions, labels)
    smoke_loss.backward()

    for name, parameter in (
        ("quantum_ry", model.quantum_ry),
        ("quantum_rz", model.quantum_rz),
        ("angle_projection", model.angle_projection.weight),
    ):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Gradient check failed for {name}.")

    print("Forward-pass and gradient checks: PASSED")
    print("Smoke-test loss:", round(smoke_loss.item(), 6))
    optimizer.zero_grad(set_to_none=True)

    # Prevent the diagnostic smoke test from changing training randomness.
    baseline.set_seed(SEED + 1)

    best_val_mae = float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []

    checkpoint_path = RESULT_DIR / "quantumprot_hybrid_best.pt"

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

            n = labels.size(0)
            total_loss += loss.item() * n
            seen += n

        train_mse = total_loss / seen
        val_metrics = baseline.evaluate(model, valid_loader, loss_fn)

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
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "validation_MAE": best_val_mae,
                "config": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "n_qubits": N_QUBITS,
                    "n_quantum_layers": N_Q_LAYERS,
                    "learning_rate": LEARNING_RATE,
                    "training_protocol": (
                        "Exact sequence and normalized parent controlled"
                    ),
                },
            }, checkpoint_path)
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Validation metrics are reported on the same split used for checkpoint selection; they are not an independent test estimate.
    test_metrics = baseline.evaluate(model, test_loader, loss_fn)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if DEVICE.type == "cuda" else None
    )

    print("\nBest epoch:", best_epoch)
    print("Best validation MAE:", round(best_val_mae, 5))
    print("Held-out validation metrics (checkpoint-selection split):", test_metrics)
    print("Peak allocated GPU memory (MiB):", peak_memory_mb)

    with (RESULT_DIR / "quantumprot_hybrid_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    results = {
        "model": "QuantumProtNet-MG four-qubit simulator prototype",
        "simulator": "Differentiable PyTorch statevector",
        "device": str(DEVICE),
        "pytorch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "heldout_validation_rows_reused_for_reporting": len(test_rows),
        "trainable_parameters": parameter_count,
        "n_qubits": N_QUBITS,
        "n_quantum_layers": N_Q_LAYERS,
        "best_epoch": best_epoch,
        "best_validation_MAE": best_val_mae,
        "heldout_validation_metrics_reused_for_reporting": test_metrics,
        "train_test_exact_sequence_overlap": 0,
        "train_test_normalized_parent_overlap": 0,
        "peak_gpu_memory_MiB": peak_memory_mb,
        "seed": SEED,
        "note": (
            "Simulator-only experimental prototype. No quantum hardware "
            "or quantum advantage is claimed. Compare against the classical "
            "model trained under the same filtered-data protocol."
        ),
    }

    with (RESULT_DIR / "quantumprot_hybrid_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nSaved checkpoint:", checkpoint_path)
    print("Saved history:", RESULT_DIR / "quantumprot_hybrid_history.csv")
    print("Saved metrics:", RESULT_DIR / "quantumprot_hybrid_metrics.json")


if __name__ == "__main__":
    main()
