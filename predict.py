import cv2
import mediapipe as mp
import numpy as np
import joblib


# Configuration

GESTURES = ["hello", "bye", "thanks", "please", "yes", "no", "help", "sorry"]
MODEL_PATH = "svm_model.pkl"


# Load trained model

model = joblib.load(MODEL_PATH)
print(" Model loaded successfully")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure we have 63 values
            if len(landmarks) == 63:
                landmarks_np = np.array(landmarks).reshape(1, -1)
                pred = model.predict(landmarks_np)
                gesture_name = GESTURES[pred[0]]

                # Display gesture
                cv2.putText(frame, f'Gesture: {gesture_name}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
