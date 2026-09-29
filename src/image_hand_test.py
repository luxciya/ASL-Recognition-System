import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Load ONE image from Kaggle dataset
image_path = "asl_alphabet_train/asl_alphabet_train/A/A1.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image not found ❌")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = hands.process(image_rgb)

# Prepare fixed-size landmark arrays
left_hand = [0.0] * (21 * 3)
right_hand = [0.0] * (21 * 3)

if results.multi_hand_landmarks:
    print("Hand detected ✅")

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # Identify left or right hand
        hand_label = results.multi_handedness[idx].classification[0].label

        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        if hand_label == "Left":
            left_hand = coords
        else:
            right_hand = coords

        # Draw landmarks
        mp_draw.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

else:
    print("No hand detected ❌")

# Combine left + right hand → 126 values
landmark_data = left_hand + right_hand

print("Total landmark values:", len(landmark_data))
print("First 10 values:", landmark_data[:10])

# Show image
cv2.imshow("Kaggle Image Hand Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
