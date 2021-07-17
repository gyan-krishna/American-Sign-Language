# -*- coding: utf-8 -*-
"""
----------Phenix Labs----------
Created on Sun Jun 20 21:37:35 2021
@author: Gyan Krishna

Topic: ASL Translator
"""

import tkinter as tk
from tkinter import ttk
from tkinter.font import BOLD, Font, ITALIC
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

class asl_app:
    def __init__(self, root):
        self.root = root
        self.labels = ['A', 'B', 'C', 'D', 'E',
                       'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O',
                       'P', 'Q', 'R', 'S', 'T',
                       'U', 'V', 'W', 'X', 'Y',
                       'Z']

        self.model = keras.models.load_model('baseline.model')

        # Defining standard fonts to be used in the app
        self.small_font = Font(self.root, family = "Helvetica", size = 13, weight = BOLD, slant = ITALIC)
        self.header_font = Font(self.root, size = 25, weight = BOLD, slant = ITALIC)
        self.micro_font = Font(self.root, family = "Helvetica", size = 10, weight = BOLD, slant = ITALIC)


        #Header of the app
        self.header_label = tk.Label(self.root,
                                     text = "ASL Deep Learning Translator",
                                     font = self.header_font)
        self.header_label.grid(row = 0,
                               column = 0,
                               columnspan = 3,
                               padx = 20,
                               pady = 10)


        # Video feed
        self.camera = CamUtils(0)
        self.vid_canvas = tk.Canvas(self.root,
                                    width = self.camera.cam_width,
                                    height = self.camera.cam_height)
        self.vid_canvas.grid(row = 1,
                             column = 0,
                             padx = 10,
                             pady = 20,
                             columnspan = 3)


        # Prediction history!
        self.text_feed_label = tk.Label(self.root, text = "Text : ",
                                   font = self.small_font)
        self.text_feed_label.grid(row = 2,
                             column = 0,
                             padx = 10,
                             pady = 10,
                             sticky = tk.W)

        self.text_feed_data = tk.Label(self.root,
                                  text = "ABCDEFGHIJKL",
                                  font = self.small_font)
        self.text_feed_data.grid(row = 2,
                            column = 1,
                            padx = 10,
                            pady = 10,
                            sticky = tk.W)


        # Prediction Accuracy Bar
        self.accuracy = tk.IntVar()
        self.accuracy_label = tk.Label(self.root, text = "Accuracy : ",
                                   font = self.small_font)
        self.accuracy_label.grid(row = 3,
                             column = 0,
                             padx = 10,
                             pady = 10,
                             sticky = tk.W)

        self.accuracy_score = ttk.Progressbar(self.root,
                                              maximum = 100,
                                              length = 200,
                                              variable = self.accuracy)
        self.accuracy_score.grid(row = 3,
                                 column = 1,
                                 padx = 10,
                                 pady = 10,
                                 sticky = tk.W)


        # Current prediction Label
        self.prediction_var = tk.StringVar()
        self.prediction_var.set("A")
        self.prediction_label = tk.Label(self.root,
                                         textvariable  = self.prediction_var,
                                         height = 2,
                                         width = 4,
                                         font = self.header_font)
        self.prediction_label.grid(row = 2,
                                   column = 2,
                                   rowspan = 2,
                                   padx = 10,
                                   pady = 10)

        # Check Button for flip camera
        self.flipVal = tk.IntVar()
        self.flip_check_box = tk.Checkbutton(root,
                                             text = "Mirror Display",
                                             font = self.small_font,
                                             variable = self.flipVal)
        self.flip_check_box.grid(row = 4,
                                 column = 0,
                                 padx = 10,
                                 pady = 10,
                                 sticky = tk.W)

        # Time
        self.currTime = tk.StringVar()
        self.time_label = tk.Label(root,
                                   textvariable = self.currTime,
                                   font = self.micro_font)
        self.time_label.grid(row = 4,
                  column = 2,
                  padx = 10,
                  pady = 10)

        self.update_frame()
        self.root.mainloop()



    def get_prediction(self, landmarks):
        landmarks = landmarks.reshape(1, 3, 21, 1)
        result = self.model.predict(landmarks)
        return result

    def update_frame(self):
        ret1, frame = self.camera.cap.read()

        t = time.ctime(time.time())
        self.currTime.set( str(t) )

        if(self.flipVal.get() == 1):
            frame = cv2.flip(frame,1)

        if ret1:
            ret, landmarks = self.camera.get_landmarks(frame)
            if ret:
                res = self.get_prediction(landmarks)
                cat = self.labels[np.argmax(res)]
                acc = res[0][np.argmax(res)] * 100

                self.accuracy.set(acc)
                self.prediction_var.set(cat)
            #else:
            #    self.accuracy.set(0)
            #    self.prediction_var.set("NA")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if(ret1):
                self.img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.vid_canvas.create_image(0,
                                             0,
                                             image = self.img,
                                             anchor = tk.NW)

        self.root.after(self.camera.refresh, self.update_frame)

    def __del__(self):
        del self.camera
        self.cap.release()

class CamUtils:

    def __init__(self, cam_num):
        self.drawingModule = mediapipe.solutions.drawing_utils
        self.handsModule = mediapipe.solutions.hands

        self.cap = cv2.VideoCapture(cam_num)
        self.cam_width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.refresh = 1

    def get_landmarks(self, img):
        with self.handsModule.Hands(static_image_mode=False,
                                    min_detection_confidence=0.65,
                                    min_tracking_confidence=0.65,
                                    max_num_hands=1) as hands:

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    points = []
                    for point in self.handsModule.HandLandmark:
                        normalizedLandmark =  handLandmarks.landmark[point]
                        pt = [normalizedLandmark.x,
                              normalizedLandmark.y,
                              normalizedLandmark.z]

                        points.append(pt)
                    return True, np.array(points, dtype = 'float64')
            else:
                return False, None

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.wm_title("ASL Translator")
    app = asl_app(root)
    del app