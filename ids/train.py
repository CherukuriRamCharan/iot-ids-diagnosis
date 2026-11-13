"""Train + evaluate the two-head LSTM.

    python -m ids.train --seeds 5 --epochs 60

Split is BY NETWORK SESSION (default: train 0-6, val 7, test 8-9), so the test set contains
network conditions never seen in training. Results are averaged over several seeds
(mean/std, like the base paper's repeated-run tables).
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .model import TwoHeadLSTM


def load(data_dir):
    d = np.load(os.path.join(data_dir, "motion_sensor.npz"))
    labels = json.load(open(os.path.join(data_dir, "labels.json")))
    return d, labels


def split(session, train_s, val_s, test_s):
    return [np.where(np.isin(session, s))[0] for s in (train_s, val_s, test_s)]


def tensors(d, idx):
    return (torch.from_numpy(d["tok"][idx]), torch.from_numpy(d["cont"][idx]),
            torch.from_numpy(d["y_diag"][idx]), torch.from_numpy(d["y_loc"][idx]))


@torch.no_grad()
def predict(model, ds, bs=512):
    model.eval()
    pd, pl = [], []
    for tok, cont, _, _ in DataLoader(ds, batch_size=bs):
        a, b = model(tok, cont)
        pd.append(a.argmax(1)); pl.append(b.argmax(1))
    return torch.cat(pd).numpy(), torch.cat(pl).numpy()


def metrics(y, p):
    return {"f1": f1_score(y, p, average="macro", zero_division=0),
            "precision": precision_score(y, p, average="macro", zero_division=0),
            "recall": recall_score(y, p, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p)}


def run_seed(seed, d, labels, splits, epochs, lr, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = (TensorDataset(*tensors(d, i)) for i in splits)
    model = TwoHeadLSTM(len(labels["vocab"]), len(labels["diag"]), len(labels["loc"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for tok, cont, yd, yl in DataLoader(tr, batch_size=bs, shuffle=True):
            opt.zero_grad()
            a, b = model(tok, cont)
            (ce(a, yd) + ce(b, yl)).backward()
            opt.step()
        pd, pl = predict(model, va)
        score = f1_score(va.tensors[3].numpy(), pl, average="macro") + f1_score(va.tensors[2].numpy(), pd, average="macro")
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pd, pl = predict(model, te)
    yd, yl = te.tensors[2].numpy(), te.tensors[3].numpy()
    # consistency: does the predicted fault location belong to the predicted attack class?
    loc2diag = {}
    for a, b in zip(d["y_diag"], d["y_loc"]):
        loc2diag[int(b)] = int(a)
    consistent = float(np.mean([loc2diag[int(l)] == int(g) for l, g in zip(pl, pd)]))
    return {"diag": metrics(yd, pd), "loc": metrics(yl, pl), "consistency": consistent}, (yd, pd, yl, pl), model


def plot_cm(y, p, names, title, path):
    cm = confusion_matrix(y, p, labels=range(len(names)))
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(names) + 3),) * 2)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="features/data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--train-sessions", type=int, nargs="+", default=list(range(0, 7)))
    ap.add_argument("--val-sessions", type=int, nargs="+", default=[7])
    ap.add_argument("--test-sessions", type=int, nargs="+", default=[8, 9])
    a = ap.parse_args()

    d, labels = load(a.data)
    splits = split(d["session"], a.train_sessions, a.val_sessions, a.test_sessions)
    print(f"train {len(splits[0])} | val {len(splits[1])} | test {len(splits[2])} samples (split by session)")
    os.makedirs(a.out, exist_ok=True)

    all_res = []
    for s in range(a.seeds):
        res, preds, model = run_seed(s, d, labels, splits, a.epochs, a.lr, a.bs)
        all_res.append(res)
        print(f"seed {s}: diag F1 {res['diag']['f1']:.4f} | loc F1 {res['loc']['f1']:.4f} | consistency {res['consistency']:.4f}")
        if s == 0:
            yd, pd, yl, pl = preds
            plot_cm(yd, pd, labels["diag"], "Attack diagnosis (test sessions)", f"{a.out}/confusion_diag.png")
            plot_cm(yl, pl, labels["loc"], "Fault localization (test sessions)", f"{a.out}/confusion_loc.png")
            torch.save(model.state_dict(), f"{a.out}/model_seed0.pt")
            idx = splits[2]
            with open(f"{a.out}/test_predictions.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["row", "session", "true_diag", "pred_diag", "true_loc", "pred_loc"])
                for k, (i, tdg, pdg, tl, ploc) in enumerate(zip(idx, yd, pd, yl, pl)):
                    w.writerow([int(i), int(d["session"][i]), labels["diag"][tdg], labels["diag"][pdg],
                                labels["loc"][tl], labels["loc"][ploc]])

    summary = {}
    for task in ("diag", "loc"):
        summary[task] = {m: {"mean": float(np.mean([r[task][m] for r in all_res])),
                             "std": float(np.std([r[task][m] for r in all_res]))}
                         for m in ("f1", "precision", "recall", "accuracy")}
    summary["consistency"] = {"mean": float(np.mean([r["consistency"] for r in all_res])),
                              "std": float(np.std([r["consistency"] for r in all_res]))}
    json.dump({"per_seed": all_res, "summary": summary, "args": vars(a)}, open(f"{a.out}/metrics.json", "w"), indent=1)
    print("\n=== mean ± std over", a.seeds, "seeds (test sessions) ===")
    for task in ("diag", "loc"):
        print(task.upper(), " ".join(f"{m} {v['mean']:.4f}±{v['std']:.4f}" for m, v in summary[task].items()))
    print("consistency", f"{summary['consistency']['mean']:.4f}±{summary['consistency']['std']:.4f}")


if __name__ == "__main__":
    main()
