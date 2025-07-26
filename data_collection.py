
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create a directory for data if it doesn't exist
data_dir = 'gesture_data'
os.makedirs(data_dir, exist_ok=True)
csv_file_path = os.path.join(data_dir, 'gesture_data.csv')

# Check if CSV file exists, if not, create with header
if not os.path.exists(csv_file_path):
    # Create header for 21 landmarks * 2 coordinates (x, y) + 1 for label
    header = [f'lm_{i}_{axis}' for i in range(21) for axis in ['x', 'y']] + ['label']
    df = pd.DataFrame(columns=header)
    df.to_csv(csv_file_path, index=False)
    print(f"Created new CSV file: {csv_file_path}")
else:
    print(f"Appending to existing CSV file: {csv_file_path}")

# Function to normalize and flatten landmarks
def normalize_and_flatten_landmarks(landmarks):
    if not landmarks:
        return None

    # Get wrist landmark (landmark 0)
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y

    normalized_landmarks = []
    for lm in landmarks:
        # Normalize relative to wrist
        normalized_landmarks.append(lm.x - wrist_x)
        normalized_landmarks.append(lm.y - wrist_y)
    return np.array(normalized_landmarks).flatten()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- Gesture Data Collection ---")
print("Press 'q' to quit.")
print("Enter the label for the gesture you are about to capture (e.g., 'fist', 'open_hand', 'pointing_up'):")
print("Then, press 's' to save a frame's landmark data for the current gesture.")

current_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural selfie-view display
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize landmarks
            landmarks_list = [lm for lm in hand_landmarks.landmark]
            flattened_landmarks = normalize_and_flatten_landmarks(landmarks_list)

            if flattened_landmarks is not None:
                # Display current label and instructions
                cv2.putText(frame, f"Current Gesture: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and current_label:
                    # Append data to CSV
                    data_row = np.append(flattened_landmarks, current_label)
                    df_to_append = pd.DataFrame([data_row], columns=header)
                    df_to_append.to_csv(csv_file_path, mode='a', header=False, index=False)
                    print(f"Saved data for gesture: {current_label}")
                elif key == ord('q'):
                    break
                elif key != 255: # Any other key pressed
                    # Prompt for new label
                    print("Enter new label for gesture:")
                    current_label = input("Label: ").strip()
                    if not current_label:
                        print("Label cannot be empty. Please enter a label.")

    cv2.imshow('Hand Gesture Data Collection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")


