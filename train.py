import argparse, time, json
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--data", default="wm811k_64.npz")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--subset", type=int, default=0, help="use N samples for a quick test")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

d = np.load(a.data, allow_pickle=True)
X, y = d["X"], d["y"]
classes = sorted(set(y)); c2i = {c: i for i, c in enumerate(classes)}
y = np.array([c2i[c] for c in y])
if a.subset:
    idx = np.random.RandomState(0).permutation(len(X))[:a.subset]; X, y = X[idx], y[idx]

Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=0)

def T(x): return torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 2.0
Xtr, Xva, Xte = T(Xtr), T(Xva), T(Xte)
ytr, yva, yte = map(torch.tensor, (ytr, yva, yte))

def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
model = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128),
                      nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
                      nn.Linear(128, len(classes))).to(dev)

# sqrt-inverse-frequency weights: softer than full inverse, avoids over-boosting tiny classes
cnt = np.bincount(ytr.numpy(), minlength=len(classes))
w = torch.tensor((cnt.max() / cnt) ** 0.5, dtype=torch.float32).to(dev)
loss_fn = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), 1e-3)

def augment(x):  # rotations/flips are valid for wafer maps
    k = np.random.randint(4); x = torch.rot90(x, k, (2, 3))
    return torch.flip(x, (3,)) if np.random.rand() < 0.5 else x

def predict(x):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            out.append(model(x[i:i+1024].to(dev)).argmax(1).cpu())
    return torch.cat(out)

best, t0 = 0, time.time()
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), a.bs):
        b = perm[i:i+a.bs]
        xb, yb = augment(Xtr[b]).to(dev), ytr[b].to(dev)
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    pv = predict(Xva)
    macro = classification_report(yva, pv, output_dict=True, zero_division=0)["macro avg"]["f1-score"]
    print(f"epoch {ep+1}/{a.epochs}  val macro-F1 {macro:.3f}  elapsed {time.time()-t0:.0f}s")
    if macro > best:
        best = macro; torch.save(model.state_dict(), "wafer_cnn.pt")
train_time = time.time() - t0

model.load_state_dict(torch.load("wafer_cnn.pt"))
pt = predict(Xte)
print(classification_report(yte, pt, target_names=classes, zero_division=0))
print(confusion_matrix(yte, pt))

# inference latency (single wafer)
model.eval(); x1 = Xte[:1].to(dev)
with torch.no_grad():
    for _ in range(20): model(x1)
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(200): model(x1)
    if dev == "cuda": torch.cuda.synchronize()
lat = (time.time() - t) / 200 * 1000
print(f"train time {train_time:.0f}s, latency {lat:.2f} ms/wafer")
json.dump({"classes": classes, "train_seconds": train_time, "latency_ms": lat, "best_val_macro_f1": best},
          open("metrics.json", "w"), indent=2)
