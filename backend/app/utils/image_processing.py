from io import BytesIO
from PIL import Image

def predict_to_image(results, model):
    """
    Run prediction on an image (path or bytes) and return the annotated image bytes and list of (class, confidence).
    """
    # gather detected classes
    detections = []
    
    if len(results) > 0 and hasattr(results[0], "boxes"):
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf.item())
            detections.append((class_name, confidence))

    # render annotated image
    imb_bgr = results[0].plot()
    img_rgb = Image.fromarray(imb_bgr[..., ::-1])
    buf = BytesIO()
    img_rgb.save(buf, format="JPEG")
    annotated_bytes = buf.getvalue()

    return annotated_bytes
