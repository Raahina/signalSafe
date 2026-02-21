# app.py
import base64
import cv2
import mediapipe as mp
import numpy as np
import joblib

from mediapipe.tasks.python import vision
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory

MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "distress_classifier.joblib"

# Load trained classifier once
clf = joblib.load(CLASSIFIER_PATH)

# Set up MediaPipe hand landmarker once (IMAGE mode)
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

def extract_landmark_vector(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32).reshape(1, -1)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "no image"}), 400

    # Expect a data URL like "data:image/jpeg;base64,AAAA..."
    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(image_data)
    except Exception:
        return jsonify({"error": "bad base64"}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "bad image"}), 400

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = hand_landmarker.detect(mp_image)

    distress_prob = 0.0
    distress = False

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        vec = extract_landmark_vector(hand_landmarks)
        probs = clf.predict_proba(vec)[0]
        distress_prob = float(probs[1])
        # simple threshold for web API (tune if needed)
        distress = distress_prob > 0.6

    return jsonify({
        "distress": distress,
        "prob": distress_prob
    })

if __name__ == "__main__":
    app.run(debug=True)

