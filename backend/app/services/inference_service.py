# services/inference_service.py

import backend.app.utils.image_processing as img_proc
import os

def run_inference(model, output_dir, folder_path):
    print(f"Running inference ...")
    os.makedirs(output_dir, exist_ok=True)
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    first_image_path = os.path.join(folder_path, image_files[0])
    prediction = model.predict(first_image_path, conf=0.2, save=False)
    annotated = img_proc.predict_to_image(prediction, model)
    
    out_path = output_dir+"output.jpg"
    with open(out_path, "wb") as f:
        f.write(annotated)
    # shutil.rmtree("backend/uploads")
    return True
