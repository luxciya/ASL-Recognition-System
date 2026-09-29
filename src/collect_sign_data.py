import cv2
import mediapipe as mp
import numpy as np
import csv
import os

SIGN_LABEL = "HELLO" # change per run
SAVE_FILE = "sign_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract(hand):
    return [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]

file_exists = os.path.exists(SAVE_FILE)

with open(SAVE_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks and res.multi_handedness:
            left, right = None, None
            for i, h in enumerate(res.multi_handedness):
                if h.classification[0].label == "Left":
                    left = res.multi_hand_landmarks[i]
                else:
                    right = res.multi_hand_landmarks[i]

            row = []
            row += extract(left) if left else [0]*63
            row += extract(right) if right else [0]*63
            row.append(SIGN_LABEL)

            writer.writerow(row)
            print("Saved", SIGN_LABEL)

            for h in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collect Sign Data", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
