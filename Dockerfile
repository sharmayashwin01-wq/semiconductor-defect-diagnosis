# ROCm PyTorch base image, so the app runs on AMD GPUs.
# Check hub.docker.com/r/rocm/pytorch for a current tag if this one is unavailable.
FROM rocm/pytorch:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py app.py ./
# Model files are copied in, or mounted at runtime with -v
COPY wafer_cnn.pt metrics.json ./

ENV LLM_NAME=Qwen/Qwen2.5-7B-Instruct
ENV USE_LLM=1
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Run on an AMD GPU:
# docker run --device=/dev/kfd --device=/dev/dri --group-add video -p 8501:8501 wafer-copilot
