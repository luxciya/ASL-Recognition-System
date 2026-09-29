import cv2, time
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import pyttsx3
from collections import deque, Counter

model = tf.keras.models.load_model("models/sign_model.keras", compile=False)
encoder = joblib.load("models/sign_encoder.pkl")
scaler = joblib.load("models/sign_scaler.pkl")

engine = pyttsx3.init()
engine.setProperty("rate", 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

buffer = deque(maxlen=15)
CONF = 0.7
last_time = 0
DELAY = 1.2

def extract(hand):
    return [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]

cap = cv2.VideoCapture(0)

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

        feats = []
        feats += extract(left) if left else [0]*63
        feats += extract(right) if right else [0]*63

        X = scaler.transform([feats])
        pred = model.predict(X, verbose=0)
        conf = np.max(pred)
        label = encoder.inverse_transform([np.argmax(pred)])[0]

        if conf > CONF:
            buffer.append(label)

        if len(buffer) == buffer.maxlen:
            final = Counter(buffer).most_common(1)[0][0]
            if time.time() - last_time > DELAY:
                engine.say(final)
                engine.runAndWait()
                last_time = time.time()
                buffer.clear()

        for h in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, label, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
