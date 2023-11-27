import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import string

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', 
                    type=str, 
                    choices=['elu', 'exponential', 'relu', 'selu', 'sigmoid', 'softplus', 'softsign', 'tanh'], 
                    default='relu')
args = parser.parse_args()

model = load_model("models/" + args.model + "/smnist.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

print(label_mapping)

padding = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bounding_box = cv2.boundingRect(np.array([[(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]]).astype(np.int32))
            x, y, w, h = bounding_box
            if x >= 0 and y >= 0:
                padding_x = padding
                padding_y = padding
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w += 2 * padding_x
                h += 2 * padding_y
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                model_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                model_frame = model_frame[y:y + h, x:x + w]
                model_frame = cv2.resize(model_frame, (28, 28))
                model_frame = pd.DataFrame(model_frame)
                model_frame = (model_frame.values / 255.0).reshape(-1, 28, 28, 1)
                predicted_sign = model.predict(model_frame)
                predicted_label = np.argmax(predicted_sign)
                top3_labels = np.argsort(predicted_sign, axis=1)[0][-3:][::-1]
                top3_values = np.take(predicted_sign, top3_labels)
                top3_predictions = [(label_mapping.get(label, "Unknown"), value) for label, value in zip(top3_labels, top3_values)]                
                print("Top 3 Predictions:", top3_predictions)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
