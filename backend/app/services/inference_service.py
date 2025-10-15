# services/inference_service.py
from PIL import Image
import numpy as np
import io

def run_inference(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Preprocess image as needed for your model, e.g. resize, normalize
    im_arr = np.array(image)
    print(im_arr.shape)
    # Dummy prediction, replace with model inference logic
    prediction = model(im_arr)  # adapt for your model
    return prediction