from flask import Flask
import os

from .routes import bp


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.register_blueprint(bp)


if __name__ == "__main__":
	# run development server
	app.run(host="0.0.0.0", port=5000, debug=True)
