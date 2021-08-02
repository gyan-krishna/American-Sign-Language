# -*- coding: utf-8 -*-
"""
----------Phenix Labs----------
Created on Mon Jun 21 00:41:19 2021
@author: Gyan Krishna

Topic:
"""

import cv2
import mediapipe
import numpy as np
import tensorflow as tf
from tensorflow import keras

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

def get_landmarks(img):
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.65, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                points = []
                for point in handsModule.HandLandmark:
                    normalizedLandmark =  handLandmarks.landmark[point]
                    pt = [normalizedLandmark.x, normalizedLandmark.y, normalizedLandmark.z]
                    points.append(pt)
                return True, np.array(points, dtype = 'float64')
        else:
            return False, None

cap = cv2.VideoCapture(0)
model = keras.models.load_model('baseline.model')

while True:
    ret, frame = cap.read()
    ret, landmarks = get_landmarks(frame)

    if ret:
        result = model.predict(landmarks)
        print(result)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if(key == 27):
        break
cv2.destroyAllWindows()
cap.release()

























