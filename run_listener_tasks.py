# run_listener_tasks.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import time

from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "distress_classifier.joblib"
CONFIG_PATH = "config.json"
LOG_FILE = "events_log.jsonl"

# Load trained classifier
clf = joblib.load(CLASSIFIER_PATH)

def log_event(event_type, **kwargs):
    """Append a JSON line to the log file with timestamp and extra fields."""
    entry = {
        "ts": time.time(),
        "type": event_type,
    }
    entry.update(kwargs)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def extract_landmark_vector(hand_landmarks):
    """Flatten 21 hand landmarks (x,y,z) into a 63-dim vector."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32).reshape(1, -1)

def main():
    # Open camera (Continuity Camera via AVFoundation)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Starting listener...")

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,  # sync mode
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Default thresholds
    DISTRESS_PROB_THRESHOLD = 0.50
    DISTRESS_THRESHOLD_FRAMES = 5  # number of high-prob frames in a row

    # Load calibrated threshold if config.json exists
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                if "distress_prob_threshold" in config:
                    DISTRESS_PROB_THRESHOLD = float(config["distress_prob_threshold"])
                    print(f"Loaded calibrated threshold: {DISTRESS_PROB_THRESHOLD:.3f}")
        except Exception as e:
            print("Warning: could not load config.json:", e)

    consecutive_distress_frames = 0
    in_distress_prompt = False

    with vision.HandLandmarker.create_from_options(hand_options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(mp_image)

            distress_detected = False
            distress_prob = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                vec = extract_landmark_vector(hand_landmarks)

                # Probability of class 1 (distress)
                probs = clf.predict_proba(vec)[0]      # [p(class0), p(class1)]
                distress_prob = float(probs[1])

                if distress_prob > DISTRESS_PROB_THRESHOLD:
                    distress_detected = True

                # Draw landmarks (red if distress, green otherwise)
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    color = (0, 0, 255) if distress_detected else (0, 255, 0)
                    cv2.circle(frame, (x, y), 3, color, -1)

            # Detection state machine
            if not in_distress_prompt:
                if distress_detected:
                    consecutive_distress_frames += 1
                else:
                    consecutive_distress_frames = 0

                if consecutive_distress_frames >= DISTRESS_THRESHOLD_FRAMES:
                    consecutive_distress_frames = 0
                    in_distress_prompt = True
                    print("[INFO] Distress gesture detected. Waiting for user confirmation (A/D)...")
                    log_event("trigger", prob=distress_prob)
            else:
                # On-screen overlay asking for confirmation
                h, w, _ = frame.shape
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, int(h * 0.3)), (w, int(h * 0.7)), (0, 0, 0), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(
                    frame,
                    "DISTRESS DETECTED",
                    (int(w * 0.1), int(h * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )
                cv2.putText(
                    frame,
                    "Press A to CONFIRM, D to dismiss",
                    (int(w * 0.05), int(h * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # Status + probability display
            status_text = "Listening for distress signal (q to quit)"
            cv2.putText(
                frame,
                status_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"p(distress)={distress_prob:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Log this frame
            log_event(
                "frame",
                prob=distress_prob,
                distress_detected=bool(distress_detected),
                in_prompt=bool(in_distress_prompt)
            )

            cv2.imshow("Crisis Hand-Signal Listener", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if in_distress_prompt:
                if key == ord('a'):
                    print("[ALERT] User confirmed distress. (Here you could send SMS/email/etc.)")
                    log_event("user_confirm", prob=distress_prob)
                    in_distress_prompt = False
                elif key == ord('d'):
                    print("[INFO] User dismissed distress prompt.")
                    log_event("user_dismiss", prob=distress_prob)
                    in_distress_prompt = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
