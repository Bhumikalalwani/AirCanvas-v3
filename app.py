import base64
import cv2
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from aircanvas_core import process_frame
from flask import send_from_directory

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process_frame", methods=["POST"])
def process_frame_api():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame"}), 400

    frame_bytes = base64.b64decode(data["frame"].split(",")[1])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process → get both outputs
    camera_frame, canvas_frame = process_frame(frame)

    _, cam_buf = cv2.imencode(".jpg", camera_frame)
    _, canv_buf = cv2.imencode(".jpg", canvas_frame)

    return jsonify({
        "camera_frame": base64.b64encode(cam_buf).decode("utf-8"),
        "canvas_frame": base64.b64encode(canv_buf).decode("utf-8")
    })

if __name__ == "__main__":
    app.run(debug=True)
