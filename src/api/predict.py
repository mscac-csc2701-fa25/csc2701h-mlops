from PIL import Image
from ultralytics import YOLO

# load model
model = YOLO("src/models/fire_smoke.pt")

# load image
img_path = "data/processed/val/images/PublicDataset00510.jpg"

# perform prediction
results = model.predict(source=img_path, conf=0.25, save=True)

imb_bgr = results[0].plot()
img_rgb = Image.fromarray(imb_bgr[..., ::-1])
img_rgb.show()