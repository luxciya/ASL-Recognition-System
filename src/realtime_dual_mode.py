import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import pyttsx3
from collections import deque, Counter

# ================= LOAD MODELS =================
letter_model = tf.keras.models.load_model("models/letter_model.keras", compile=False)
letter_encoder = joblib.load("models/letter_encoder.pkl")
letter_scaler = joblib.load("models/letter_scaler.pkl")

sign_model = tf.keras.models.load_model("models/sign_model.keras", compile=False)
sign_encoder = joblib.load("models/sign_encoder.pkl")
sign_scaler = joblib.load("models/sign_scaler.pkl")

print("✅ Letter & Sign models loaded")

# ================= SPEECH =================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= MODES =================
MODE = "LETTER"   # LETTER / SIGN

letter_buffer = deque(maxlen=12)
sign_buffer = deque(maxlen=15)

CONF_LETTER = 0.65
CONF_SIGN = 0.7

STABLE_LETTER = 7
STABLE_SIGN = 10

sentence = ""
last_letter_emit = 0
last_sign_emit = 0
EMIT_DELAY = 1.0

current_label = ""
current_conf = 0.0
last_emitted = ""

# ================= FUNCTIONS =================
def extract(hand):
    return [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]

def get_stable(buffer, count):
    if len(buffer) < count:
        return None
    label, freq = Counter(buffer).most_common(1)[0]
    return label if freq >= count else None

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("🎥 L = Letter | S = Sign | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        left, right = None, None

        # ---- Correct Left / Right Ordering ----
        for i, info in enumerate(results.multi_handedness):
            if info.classification[0].label == "Left":
                left = results.multi_hand_landmarks[i]
            else:
                right = results.multi_hand_landmarks[i]

        features = []
        features += extract(left) if left else [0] * 63
        features += extract(right) if right else [0] * 63

        # ================= LETTER MODE =================
        if MODE == "LETTER":
            X = letter_scaler.transform([features])
            preds = letter_model.predict(X, verbose=0)
            current_conf = float(np.max(preds))
            current_label = letter_encoder.inverse_transform(
                [np.argmax(preds)]
            )[0]

            if current_conf > CONF_LETTER:
                letter_buffer.append(current_label)

            stable = get_stable(letter_buffer, STABLE_LETTER)

            if stable and time.time() - last_letter_emit > EMIT_DELAY:
                if stable == "space":
                    sentence += " "
                elif stable == "del":
                    sentence = sentence[:-1]
                else:
                    sentence += stable
                    engine.say(stable)
                    engine.runAndWait()

                last_letter_emit = time.time()
                last_emitted = stable
                letter_buffer.clear()

        # ================= SIGN MODE =================
        else:
            X = sign_scaler.transform([features])
            preds = sign_model.predict(X, verbose=0)
            current_conf = float(np.max(preds))
            current_label = sign_encoder.inverse_transform(
                [np.argmax(preds)]
            )[0]

            if current_conf > CONF_SIGN:
                sign_buffer.append(current_label)

            stable = get_stable(sign_buffer, STABLE_SIGN)

            if stable and time.time() - last_sign_emit > EMIT_DELAY:
                engine.say(stable)
                engine.runAndWait()
                last_sign_emit = time.time()
                last_emitted = stable
                sign_buffer.clear()

        # Draw landmarks
        for h in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

    # ================= UI =================
    cv2.putText(frame, f"MODE: {MODE}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.putText(frame, f"Prediction: {current_label} ({current_conf:.2f})",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, f"Sentence: {sentence}",
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ASL Dual Mode Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        MODE = "LETTER"
        letter_buffer.clear()
        print("🔤 Letter Mode")
    elif key == ord('s'):
        MODE = "SIGN"
        sign_buffer.clear()
        print("✋ Sign Mode")

cap.release()
cv2.destroyAllWindows()
