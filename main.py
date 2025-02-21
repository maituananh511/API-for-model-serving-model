from fastapi import FastAPI, UploadFile, File
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

# Triton server URL
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"

# Connect to Triton Inference Server
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Image preprocessing function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).numpy()  # Convert to NumPy format (3, 224, 224)
    return image  # Không cần thêm batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        transformed_img = preprocess_image(image)

        # Prepare Triton input
        inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
        inputs.set_data_from_numpy(transformed_img, binary_data=True)

        # Specify the output layer
        outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

        # Query Triton Inference Server
        results = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])
        inference_output = results.as_numpy("fc6_1").astype(str)

        # Process output
        output_list = np.squeeze(inference_output)[:5].tolist()
        
        return {"predictions": output_list}

    except Exception as e:
        return {"error": str(e)}
