# collect_data_tasks.py
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

def extract_landmark_vector(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_samples = []
    all_labels = []

    print("Instructions:")
    print(" - Show your hand to the camera.")
    print(" - Press '0' to record a 'no distress' sample.")
    print(" - Press '1' to record a 'distress signal' sample.")
    print(" - Press 'q' to quit and save data.")
    print()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(mp_image)

            key = cv2.waitKey(1) & 0xFF

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                vec = extract_landmark_vector(hand_landmarks)

                if key == ord('0'):
                    all_samples.append(vec)
                    all_labels.append(0)
                    print("Captured sample for class 0 (no distress)")

                elif key == ord('1'):
                    all_samples.append(vec)
                    all_labels.append(1)
                    print("Captured sample for class 1 (distress)")

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(
                frame,
                "Press 0 (no distress), 1 (distress), q (quit)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect Hand Data", frame)

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_samples) == 0:
        print("No samples collected, nothing to save.")
        return

    X = np.stack(all_samples)
    y = np.array(all_labels, dtype=np.int64)

    np.save("X_hand.npy", X)
    np.save("y_hand.npy", y)

    print(f"Saved {len(all_samples)} samples to X_hand.npy and y_hand.npy")

if __name__ == "__main__":
    main()

