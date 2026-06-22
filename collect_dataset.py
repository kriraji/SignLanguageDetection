import cv2
import mediapipe as mp
import csv
import os

GESTURES = ["hello", "bye", "thanks", "please", "yes", "no", "help", "sorry"]
SAVE_PATH = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

for g in GESTURES:
    file_path = os.path.join(SAVE_PATH, f"{g}.csv")
    if not os.path.exists(file_path):
        open(file_path, "w").close()

cap = cv2.VideoCapture(0)
current = 0

print("Press SPACE to capture 1 sample")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {GESTURES[current]}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            with open(os.path.join(SAVE_PATH, f"{GESTURES[current]}.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)

            print(f"Captured sample for {GESTURES[current]}")

    elif key == ord('n'):  
        current = (current + 1) % len(GESTURES)
        print(f"Switched to: {GESTURES[current]}")

    elif key == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
