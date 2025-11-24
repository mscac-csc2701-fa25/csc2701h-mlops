from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import base64

from . import predict as predict_module


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))


@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	if "image" not in request.files:
		return redirect(url_for("index"))

	file = request.files["image"]
	if file.filename == "":
		return redirect(url_for("index"))

	filename = secure_filename(file.filename)
	save_path = os.path.join(UPLOAD_DIR, filename)
	file.save(save_path)

	# call predict module
	with open(save_path, "rb") as f:
		img_bytes = f.read()
	annotated_bytes, detections = predict_module.predict_image(image_bytes=img_bytes)

	# encode annotated image as base64 to embed in HTML
	b64 = base64.b64encode(annotated_bytes).decode("utf-8")
	data_uri = f"data:image/jpeg;base64,{b64}"

	return render_template("index.html", result_image=data_uri, detections=detections)


if __name__ == "__main__":
	# run development server
	app.run(host="0.0.0.0", port=5000, debug=True)
