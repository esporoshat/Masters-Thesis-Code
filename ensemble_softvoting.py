#!/usr/bin/env python
"""ESG 3 way Ensemble Classifier (deterministic, memory friendly)

Models (GPU friendly)
---------------------
* **FinBERT‑ESG**  – finance‑tuned 4‑class (drop *None*)
* **DeBERTa‑v3‑base‑MNLI** – lightweight zero‑shot NLI

Run example
-----------
```bash
python esg_ensemble.py \
    --input test_esg.csv \
    --output test_preds.csv \
    --device cuda

"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, pipeline)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
LABEL_ALIASES = {
    "environment": "Environmental",
    "environmental": "Environmental",
    "env": "Environmental",
    "governance": "Governance",
    "gov": "Governance",
    "social": "Social"
}
TEXT_COL = "Text"
LABEL_COL = "ESG_Category"
LABELS = ["Environmental", "Social", "Governance"]
DEFAULT_ZS_MODEL = "MoritzLaurer/deberta-v3-base-mnli"

# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------
def vec3_esgroberta(text, tok, mod, device):
    toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = mod(**toks).logits[0].cpu()
        probs = F.softmax(logits, dim=0)

    id2label = mod.config.id2label
    none_index = [i for i, label in id2label.items() if label.lower() == "none"]
    none_index = none_index[0] if none_index else None

    keep = {i: p for i, p in enumerate(probs) if i != none_index}
    v = torch.tensor([keep.get(0, 0.), keep.get(1, 0.), keep.get(2, 0.)])
    return F.softmax(v, dim=0)

def vec3_finbert(scores):
    keep = {d["label"]: d["score"] for d in scores if d["label"] != "None"}
    v = torch.tensor([
        keep.get("Environmental", 0.),
        keep.get("Social", 0.),
        keep.get("Governance", 0.),
    ])
    return F.softmax(v, dim=0)

def vec3_mnli(text, pipe):
    out = pipe(text, candidate_labels=LABELS, hypothesis_template="This text is about {}.")
    idx = [out["labels"].index(l) for l in LABELS]
    return torch.tensor([out["scores"][i] for i in idx])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--zs_on_cpu", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else "cuda" if torch.cuda.is_available() else "cpu")
    device_id = -1 if args.zs_on_cpu else 0

    set_seeds()

    df = pd.read_csv(args.input)
    df[LABEL_COL] = df[LABEL_COL].map(lambda x: LABEL_ALIASES.get(x.lower(), x))

    # Load models
    print("Loading FinBERT...")
    pf = pipeline("text-classification", model="yiyanghkust/finbert-esg", top_k=None, device=device_id)

    print("Loading esgroberta4...")
    tok_esgroberta = AutoTokenizer.from_pretrained("/home/u493846/saved_model/esgroberta4/esgroberta4")
    mod_esgroberta = AutoModelForSequenceClassification.from_pretrained("/home/u493846/saved_model/esgroberta4/esgroberta4").to(device)
    mod_esgroberta.eval()

    print("Loading DeBERTa MNLI...")
    pipe = pipeline("zero-shot-classification", model=DEFAULT_ZS_MODEL, device=device_id)

    finbert_preds, esgroberta_preds, mnli_preds, ensemble_preds, golds = [], [], [], [], []

    for text, label in zip(df[TEXT_COL], df[LABEL_COL]):
        f_vec = vec3_finbert(pf(text)[0])
        e_vec = vec3_esgroberta(text, tok_esgroberta, mod_esgroberta, device)
        m_vec = vec3_mnli(text, pipe)
        final_vec = (f_vec + e_vec + m_vec) / 3.0

        finbert_preds.append(torch.argmax(f_vec).item())
        esgroberta_preds.append(torch.argmax(e_vec).item())
        mnli_preds.append(torch.argmax(m_vec).item())
        ensemble_preds.append(torch.argmax(final_vec).item())
        golds.append(LABELS.index(label))

    # Save to CSV
    df_out = df.copy()
    df_out["finbert_pred"] = [LABELS[p] for p in finbert_preds]
    df_out["esgroberta_pred"] = [LABELS[p] for p in esgroberta_preds]
    df_out["mnli_pred"] = [LABELS[p] for p in mnli_preds]
    df_out["ensemble_pred"] = [LABELS[p] for p in ensemble_preds]
    df_out.to_csv(args.output, index=False)

    # Print performance
    print("\nFinBERT:")
    print(classification_report(golds, finbert_preds, target_names=LABELS))
    print("\nESGRoberta:")
    print(classification_report(golds, esgroberta_preds, target_names=LABELS))
    print("\nDeBERTa MNLI:")
    print(classification_report(golds, mnli_preds, target_names=LABELS))
    print("\nEnsemble:")
    print(classification_report(golds, ensemble_preds, target_names=LABELS))

if __name__ == "__main__":
    main()
