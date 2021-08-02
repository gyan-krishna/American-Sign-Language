# -*- coding: utf-8 -*-
"""
----------Phenix Labs----------
Created on Mon Jun 14 14:55:56 2021
@author: Gyan Krishna

Topic:
"""

import tkinter as tk
#from tkinter.ttk import Progressbar
import cv2
from PIL import Image, ImageTk

width, height = 800, 600
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

win = tk.Tk()
win.bind('<Escape>', lambda e: win.quit())

header_label = tk.Label(win,
                        text = "ASL DEEP LEARNING TRANSLATOR",
                        height = 2,
                        font=("Arial", 20))
header_label.grid(row = 0, column = 0, columnspan = 2)

lmain = tk.Label(win)
lmain.grid(row = 1, column = 0, columnspan = 2)

header_label = tk.Label(win,
                        text = "ABCDEFGHIJ",
                        height = 2,
                        font=("Arial", 15))
header_label.grid(row = 2, column = 0)

#progress = Progressbar(win,
#                       orient = tk.HORIZONTAL,
#                       length = 100,
#                       mode = 'determinate')

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
win.mainloop()
cap.release()