"""Wafer copilot pipeline: classifier -> action threshold -> retrieval -> local LLM.

Runs on an AMD GPU (ROCm build of PyTorch) when available, otherwise CPU.
All settings come from environment variables; no secrets are stored in code.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm reuses the "cuda" name
LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen2.5-7B-Instruct")
USE_LLM = os.environ.get("USE_LLM", "1") == "1"
MODEL_PATH = os.environ.get("MODEL_PATH", "wafer_cnn.pt")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.json")

# Illustrative notes written for this project (synthetic, not from a real fab).
KB = {
    "Center": "Center pattern: failing dies clustered at the wafer center. Likely causes to check: CMP polishing non-uniformity, center-heavy gas flow or temperature in deposition/etch, spin-coat issues at the center. Suggested checks: review CMP removal profile, chamber gas-flow and temperature uniformity. Urgency: medium.",
    "Donut": "Donut pattern: ring-shaped failure zone with a healthier center and edge. Likely causes to check: radial non-uniformity in etch or deposition, CMP ring effects, temperature profile across the chuck. Suggested checks: radial thickness map, chamber uniformity tuning. Urgency: medium-high.",
    "Edge-Loc": "Edge-Loc pattern: failures localized in one region at the wafer edge. Likely causes to check: wafer handling or clamping damage, edge bead or resist edge issues, localized contamination near the edge. Suggested checks: inspect clamp/robot contact points, edge bead removal step. Urgency: medium.",
    "Edge-Ring": "Edge-Ring pattern: failures in a ring along the full wafer edge. Likely causes to check: etch or deposition non-uniformity at the edge, worn or misaligned chamber edge/focus ring, resist edge-bead or edge exclusion issues. Suggested checks: chamber edge uniformity, inspect the edge/focus ring wear, review recent maintenance on the tool. Urgency: high, hold the lot for review.",
    "Loc": "Loc pattern: a localized cluster of failures away from the edge. Likely causes to check: particle contamination, local lithography defect, handling mark, probe or test issue in that region. Suggested checks: defect inspection of the region, compare with neighboring wafers in the lot. Urgency: medium.",
    "Near-full": "Near-full pattern: almost all dies fail. Likely causes to check: gross process failure, wrong recipe, upstream contamination event, test or probe setup error. Suggested checks: verify test setup first, then recipe and tool logs for the lot. Urgency: critical, hold the lot immediately.",
    "Random": "Random pattern: failures scattered across the wafer with no structure. Likely causes to check: particle contamination, chemical or ambient contamination, general tool cleanliness. Suggested checks: particle counts on the tool, cleanroom and chemical logs. Urgency: medium-high if widespread across the lot.",
    "Scratch": "Scratch pattern: thin line or arc of failing dies. Likely causes to check: wafer handling or robot contact, CMP slurry agglomerates, cassette or end-effector contact. Suggested checks: inspect handling hardware and CMP consumables, check which tool step the scratch aligns with. Urgency: high.",
    "none": "None pattern: only scattered background failures, no systematic pattern. Suggested action: no action needed, continue normal monitoring. Urgency: low.",
}


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


def build_cnn(n_classes):
    return nn.Sequential(
        _block(1, 32), _block(32, 64), _block(64, 128), _block(128, 128),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )


class Copilot:
    def __init__(self):
        self.classes = json.load(open(METRICS_PATH))["classes"]
        self.cnn = build_cnn(len(self.classes)).to(DEVICE)
        self.cnn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.cnn.eval()

        from sentence_transformers import SentenceTransformer, util
        self._util = util
        self.emb = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        self.names = list(KB)
        self.vecs = self.emb.encode([KB[n] for n in self.names], convert_to_tensor=True)

        self.tok = self.llm = None
        if USE_LLM:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(LLM_NAME)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto")

    def retrieve(self, query, k=2):
        q = self.emb.encode(query, convert_to_tensor=True)
        s = self._util.cos_sim(q, self.vecs)[0]
        return [(self.names[i], KB[self.names[i]]) for i in s.topk(k).indices.tolist()]

    @staticmethod
    def decide(label, conf):
        if label == "none":
            return "NO ACTION", "No systematic pattern detected — continue normal monitoring."
        if conf >= 0.8:
            return "HOLD LOT", f"High confidence ({conf:.0%}) defect detected — stop the lot immediately."
        if conf >= 0.5:
            return "FLAG FOR REVIEW", f"Moderate confidence ({conf:.0%}) — engineer inspection required."
        return "UNCERTAIN", f"Low confidence ({conf:.0%}) — send to manual review before any action."

    def classify(self, wafer):
        x = torch.tensor(np.asarray(wafer), dtype=torch.float32)[None, None] / 2.0
        with torch.no_grad():
            p = torch.softmax(self.cnn(x.to(DEVICE)), 1)[0].cpu()
        conf, idx = p.max(0)
        return self.classes[int(idx)], float(conf), p.numpy()

    def diagnose(self, wafer):
        t0 = time.time()
        label, conf, probs = self.classify(wafer)
        action, reason = self.decide(label, conf)
        notes = self.retrieve(f"{label} wafer map pattern", k=2)
        answer = None
        if self.llm is not None:
            context = "\n".join(f"[{n}] {t}" for n, t in notes)
            prompt = (
                "You are a fab process engineer assistant. Answer ONLY from the notes below.\n"
                f"Wafer classified as {label} with confidence {conf:.2f}. System decision: {action} ({reason})\n"
                f"Notes:\n{context}\n\n"
                "Write: 1) Summary (1 line), 2) Likely causes to check (bullets), "
                "3) Recommended action, 4) Sources (note names in brackets). "
                "Say 'likely causes to check', never state a cause as certain.")
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inp = self.tok(text, return_tensors="pt").to(DEVICE)
            out = self.llm.generate(**inp, max_new_tokens=250, do_sample=False)
            answer = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return {
            "label": label, "confidence": conf, "probs": probs, "action": action,
            "reason": reason, "notes": [n for n, _ in notes], "answer": answer,
            "seconds": time.time() - t0, "device": DEVICE,
        }
