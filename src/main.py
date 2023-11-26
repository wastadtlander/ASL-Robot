import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import string

model = load_model("models/relu/smnist.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

label_mapping = {idx: letter for idx, letter in enumerate(string.ascii_uppercase) if letter not in ['J', 'Z']}

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                model_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                model_frame = model_frame[y:y + h, x:x + w]
                model_frame = cv2.resize(model_frame, (28,28))
                model_frame = pd.DataFrame(model_frame.flatten()).T
                model_frame = model_frame.values.astype(float) / 255.0
                model_frame = model_frame.reshape(-1, 28, 28, 1)
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
