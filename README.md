# API for model serving

This repository provides a FastAPI-based API for serving ML models using NVIDIA Triton Inference Server.

## Setup
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn tritonclient[all] pillow torch torchvision
   ```
2. Run Triton Server (ensure Docker is installed):
   ```bash
   docker run --rm --gpus all -p 8000:8000 -v "D:/working/fast_api/models:/models" nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
   ```
3. Start FastAPI:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

## API Usage
- **POST /predict**: Send an image (JPEG/PNG) for inference.
- **Example Request (Python):**
   ```python
   import requests
   url = "http://127.0.0.1:8080/predict"
   files = {"file": open("image.jpg", "rb")}
   print(requests.post(url, files=files).json())
   ```

## References
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)

