# Use NVIDIA Triton SDK base image
FROM nvcr.io/nvidia/tritonserver:25.01-py3-sdk

# Set working directory
WORKDIR /workspace

# Install necessary Python packages
RUN pip install --no-cache-dir fastapi uvicorn numpy tritonclient torchvision pillow python-multipart

# Copy FastAPI app into the container
COPY main.py /workspace/main.py

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
