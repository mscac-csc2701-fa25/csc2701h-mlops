from PIL import Image
from ultralytics import YOLO
import argparse

# take path to image as command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path",
    type=str,
    help="Path to the input image",
    default="data/processed/val/images/PublicDataset00510.jpg"
)
args = parser.parse_args()

# load model
model = YOLO("src/models/fire_smoke.pt")

# load image
img_path = args.image_path

# perform prediction
results = model.predict(source=img_path, conf=0.25, save=True)

# output detected classes
print("\nPrediction results")
print("------------------")
print("class: confidence")
for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    confidence = box.conf.item()
    print(f"{class_name}: {confidence:.2f}")
print()

# display results
imb_bgr = results[0].plot()
img_rgb = Image.fromarray(imb_bgr[..., ::-1])
img_rgb.show()