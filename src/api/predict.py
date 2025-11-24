from io import BytesIO
import os
from typing import List, Tuple, Optional

from PIL import Image
from ultralytics import YOLO


# Load model once (keeps it in memory for repeated calls)
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "src/models/fire_smoke.pt")
_model = None


def _get_model(model_path: str = DEFAULT_MODEL_PATH):
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model


def predict_image(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
) -> Tuple[bytes, List[Tuple[str, float]]]:
    """Run prediction on an image (path or bytes) and return the annotated image bytes and list of (class, confidence).

    Returns:
        (annotated_image_bytes_jpeg, [(class_name, confidence), ...])
    """
    # prepare input file
    tmp_path = None
    if image_bytes is not None:
        tmp_path = "__tmp_predict_input.jpg"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        img_src = tmp_path
    elif image_path is not None:
        img_src = image_path
    else:
        raise ValueError("Either image_path or image_bytes must be provided")

    model = _get_model(model_path)

    # perform prediction without saving to disk (we'll render results ourselves)
    results = model.predict(source=img_src, conf=conf, save=False)

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

    # clean up temporary file if used
    if tmp_path is not None and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return annotated_bytes, detections


if __name__ == "__main__":
    # keep simple CLI compatibility
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--model_path", type=str, help="Path to the model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--conf", type=float, help="Confidence threshold", default=0.25)
    args = parser.parse_args()

    annotated, detections = predict_image(image_path=args.image_path, model_path=args.model_path, conf=args.conf)

    print("\nPrediction results")
    print("------------------")
    print("class: confidence")
    for name, conf_val in detections:
        print(f"{name}: {conf_val:.2f}")

    # save annotated image next to input
    if args.image_path:
        out_path = os.path.splitext(args.image_path)[0] + "_pred.jpg"
        with open(out_path, "wb") as f:
            f.write(annotated)
        print(f"Annotated image written to: {out_path}")