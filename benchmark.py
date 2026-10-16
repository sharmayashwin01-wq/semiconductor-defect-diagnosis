"""Run once on the AMD GPU machine:  python benchmark.py
Needs next to it: pipeline.py, wafer_cnn.pt, metrics.json, wm811k_64.npz
Writes results.json (the app shows it, and the README table uses it).
"""
import copy
import json
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import pipeline

cp = pipeline.Copilot()
d = np.load("wm811k_64.npz", allow_pickle=True)
X, y = d["X"], d["y"]
classes = cp.classes

# Rebuild the exact held-out test split used in train.py
yi = np.array([classes.index(c) for c in y])
idx = np.arange(len(y))
_, tmp = train_test_split(idx, test_size=0.3, stratify=yi, random_state=0)
_, test_idx = train_test_split(tmp, test_size=0.5, stratify=yi[tmp], random_state=0)

# 5 held-out wafers per class (fewer if the class is tiny)
rng = np.random.RandomState(7)
sample = []
for c in classes:
    pool = [j for j in test_idx if y[j] == c]
    sample += list(rng.choice(pool, min(5, len(pool)), replace=False))

cp.diagnose(X[sample[0]])  # warm-up, not timed
times, correct = [], 0
for i in sample:
    t = time.time()
    r = cp.diagnose(X[i])
    times.append(time.time() - t)
    correct += int(r["label"] == y[i])
print(f"{len(sample)} held-out wafers | accuracy {correct/len(sample):.2f} | "
      f"avg {np.mean(times):.2f}s per wafer end-to-end")


def classifier_latency_ms(model, dev, n=200):
    x = torch.tensor(X[test_idx[0]], dtype=torch.float32)[None, None].div(2.0).to(dev)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


res = {
    "device": pipeline.DEVICE,
    "gpu_name": torch.cuda.get_device_name(0) if pipeline.DEVICE == "cuda" else "cpu",
    "torch": torch.__version__,
    "n_wafers": len(sample),
    "sample_accuracy": correct / len(sample),
    "avg_pipeline_seconds": float(np.mean(times)),
    "classifier_ms_on_device": classifier_latency_ms(cp.cnn, pipeline.DEVICE),
    "classifier_ms_cpu": classifier_latency_ms(copy.deepcopy(cp.cnn).cpu(), "cpu"),
}

if cp.llm is not None:
    msgs = [{"role": "user", "content": "In 3 short bullets, list likely causes to check for an edge-ring wafer defect."}]
    text = cp.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = cp.tok(text, return_tensors="pt").to(pipeline.DEVICE)
    cp.llm.generate(**inp, max_new_tokens=16, do_sample=False)  # warm-up
    t = time.time()
    out = cp.llm.generate(**inp, max_new_tokens=200, do_sample=False)
    dt = time.time() - t
    n = out.shape[1] - inp["input_ids"].shape[1]
    res["llm"] = pipeline.LLM_NAME
    res["llm_tokens_per_sec"] = n / dt

json.dump(res, open("results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print("\nTake a rocm-smi screenshot WHILE this script runs (watch -n 1 rocm-smi in a second terminal).")
