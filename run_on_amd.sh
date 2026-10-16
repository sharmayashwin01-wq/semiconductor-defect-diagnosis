#!/usr/bin/env bash
# One command on the AMD machine:  bash run_on_amd.sh
# Needs in this folder: pipeline.py app.py benchmark.py requirements.txt wafer_cnn.pt metrics.json wm811k_64.npz
set -u
mkdir -p proof

echo "== 1. GPU check (idle) =="
(rocm-smi || amd-smi || echo "rocm-smi not found") 2>&1 | tee proof/rocm-smi-idle.txt
python3 -c "import torch; print('torch', torch.__version__, '| gpu available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" | tee proof/torch-check.txt

echo "== 2. Install packages (does NOT touch torch) =="
pip install -r requirements.txt

echo "== 3. Download models now (slowest step) =="
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print("models cached")
EOF

echo "== 4. Benchmark with GPU logging =="
( while true; do date; rocm-smi --showuse --showpower --showmeminfo vram 2>/dev/null; sleep 2; done ) > proof/gpu_log_during_benchmark.txt 2>&1 &
LOGGER=$!
python3 benchmark.py 2>&1 | tee proof/benchmark_output.txt
kill $LOGGER 2>/dev/null
cp results.json proof/ 2>/dev/null

echo "== 5. Done =="
echo "Evidence is in ./proof (rocm-smi logs, benchmark output, results.json)."
echo "ALSO take a screenshot of 'watch -n 1 rocm-smi' while the benchmark runs next time, or from the app under load."
echo "Start the demo app with:"
echo "  streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
