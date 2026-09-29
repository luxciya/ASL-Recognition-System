import cv2
import mediapipe as mp
import os
import csv

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# Root dataset path (double folder – Kaggle structure)
ROOT_DATASET = "asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "asl_landmarks_full.csv"

# Open CSV file
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)

    # Loop through A–Z folders
    for label in sorted(os.listdir(ROOT_DATASET)):
        folder_path = os.path.join(ROOT_DATASET, label)

        # Skip non-folder files
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing letter: {label}")

        # Loop through images in each folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Fixed-size landmark arrays
            left_hand = [0.0] * (21 * 3)
            right_hand = [0.0] * (21 * 3)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[idx].classification[0].label
                    coords = []

                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    if hand_label == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords

                # Combine landmarks + label
                row = left_hand + right_hand + [label]
                writer.writerow(row)

print("✅ Full ASL landmark dataset created successfully!")
