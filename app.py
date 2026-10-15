from pathlib import Path
import sys

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import train_quantumprot_hybrid_scaffold as hybrid

CHECKPOINT = ROOT / "quantumprot_hybrid_best.pt"

st.set_page_config(
    page_title="QuantumProtNet-MG",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 QuantumProtNet-MG")
st.subheader("Protein Stability Prediction")
st.write(
    "An experimental protein-sequence regression demo using a custom "
    "neural network and a differentiable four-qubit statevector simulator."
)

@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = hybrid.DEVICE
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = hybrid.QuantumProtHybridRegressor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device, checkpoint


with st.sidebar:
    st.header("Model information")
    st.write("Architecture: custom sequence encoder + simulator")
    st.write("Qubits: 4")
    st.write("Quantum circuit layers: 2")
    st.write("Simulator: PyTorch statevector")
    st.write("Hardware: classical CPU/GPU; no quantum hardware")

    st.caption(
        "Checkpoint: results/quantumprot_hybrid_scaffold/"
        "quantumprot_hybrid_best.pt"
    )

st.markdown("### Enter a protein sequence")
st.write(
    "Enter a sequence using the 20 standard amino-acid letters. "
    "The demo accepts sequences of 43–50 residues, matching the "
    "main sequence-length range represented in the training data."
)

sequence_input = st.text_area(
    "Amino-acid sequence",
    value="",
    placeholder="Paste a protein sequence here, e.g. ACDEFGHIK...",
    height=130,
)

if st.button("Predict stability", type="primary", use_container_width=True):
    sequence = "".join(sequence_input.split()).upper()
    allowed = set(hybrid.baseline.AMINO_ACIDS)

    if not sequence:
        st.error("Please enter a protein sequence.")
    elif set(sequence) - allowed:
        invalid = sorted(set(sequence) - allowed)
        st.error(
            "Unsupported characters: " + ", ".join(invalid)
            + ". Use only the 20 standard amino-acid letters."
        )
    elif not 43 <= len(sequence) <= 50:
        st.error(
            f"Sequence length is {len(sequence)}. "
            "This demo accepts 43–50 residues only."
        )
    else:
        try:
            with st.spinner("Loading model and calculating prediction..."):
                model, device, checkpoint = load_model()

                token_ids = [
                    hybrid.baseline.AA_TO_ID[aa] for aa in sequence
                ]
                tokens = torch.tensor(
                    [token_ids], dtype=torch.long, device=device
                )
                mask = tokens != hybrid.baseline.PAD_ID

                with torch.inference_mode():
                    prediction = model(tokens, mask).item()

            st.success("Prediction completed")
            st.metric("Predicted stability score", f"{prediction:.4f}")

            col1, col2 = st.columns(2)
            col1.metric("Sequence length", f"{len(sequence)} residues")
            col2.metric(
                "Checkpoint epoch",
                str(checkpoint.get("best_epoch", "Not recorded")),
            )

            st.caption(f"Inference device: {device}")

            st.info(
                "Interpretation: this is a model-generated regression score "
                "on the dataset's label scale. Do not interpret it as a "
                "laboratory measurement or as proof that a protein is "
                "stable or unstable without an established label definition "
                "and experimental validation."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Research prototype only. The simulator does not run on quantum "
    "hardware. No quantum advantage, clinical utility, or experimental "
    "accuracy is claimed. Performance on new proteins is not guaranteed."
)


