## Title
Wafer Defect Copilot

## Short description (one line)
A local, AMD GPU-powered assistant that classifies semiconductor wafer defects, decides whether to hold the lot, and cites a troubleshooting procedure.

## Long description
Fabs generate wafer maps faster than engineers can read them, and finding the right procedure after a defect appears is slow manual work. Wafer Defect Copilot takes a wafer map and (1) classifies the failure pattern with a CNN trained on WM-811K (172,950 labeled wafers), (2) applies action thresholds (hold lot, flag for review, uncertain, no action), (3) retrieves the matching troubleshooting note with embedding search, and (4) uses a locally hosted Qwen2.5-7B-Instruct to write a short recommendation that cites its source and phrases causes as "likely causes to check".

Everything runs on AMD hardware: training, classifier inference, embeddings and LLM generation ran on an AMD Radeon Pro W7900D with ROCm. No external LLM API is used, which also fits fabs that cannot send process data to the cloud.

Results: test accuracy 0.97, macro-F1 0.86, training in 184 s, classifier latency 0.45 ms per wafer, LLM about 30 tokens/s. Honest limits: the troubleshooting notes are illustrative, Scratch is the weakest class (F1 0.70), and Edge-Ring/Edge-Loc are sometimes confused. The ROI panel uses editable assumptions, not measured fab data.

## Tags
AMD ROCm, PyTorch, Qwen2.5, Streamlit, RAG, manufacturing, semiconductor

## Demo video script (3:00 max)
- 0:00-0:25 Problem: a night-shift process engineer gets hundreds of wafer maps. Looking at each one and searching SOPs takes about 25 minutes per case (state this as an assumption).
- 0:25-0:50 Solution and AMD: show the architecture diagram. Say it runs on an AMD Radeon Pro W7900D with ROCm, no cloud LLM API.
- 0:50-1:50 Live demo: pick a held-out Edge-Ring wafer, press Diagnose, show class, confidence, HOLD LOT action, cited recommendation, analysis time of 3-5 seconds. Then show one low-confidence case going to manual review.
- 1:50-2:20 Proof: show the rocm-smi screenshot with GPU usage above 0, and the results table (F1 0.86, 0.45 ms, 30 tokens/s, GPU vs CPU).
- 2:20-2:45 Business value: the ROI panel, saying the dollars come from editable assumptions. Mention time to diagnosis, 25 min vs 3-5 s.
- 2:45-3:00 Limits and next steps: real fab SOPs, drift view across a lot, more wafer data.

## Slide outline (8 slides)
1. Title and one-line value
2. The problem and the user (night-shift process engineer)
3. Solution flow (wafer to action)
4. Architecture diagram with AMD hardware labeled
5. AMD usage and proof (rocm-smi, ROCm PyTorch, Dockerfile)
6. Results table and confusion matrix highlights
7. Business value and ROI assumptions
8. Limits, next steps, repo and demo links

## Pre-submit checklist
- [ ] Repo public, no keys or secrets, README has "AMD Technologies Used"
- [ ] Docker builds
- [ ] App URL opens in a private window
- [ ] Video 3:00 or shorter
- [ ] Slides, cover image, title, descriptions, tags
- [ ] Submitted at least a few hours before the deadline
