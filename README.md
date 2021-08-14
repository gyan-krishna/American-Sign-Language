# American Sign Language Translator


American-Sign-Language is a deep learning model that is aimed at translating
sign language from a video stream or an image. the model runs of landmarks extracted by media pipe which is fed to a cnn model with 4 trailing dense layers. the model is trainned using keras. This is my final year project for BSc Computer Science.

## Run these commands to install necessary packages

```
!pip install mediapipe
!pip install opencv-python
!pip install tensorflow
!pip install keras
!pip install tqdm
!pip install scikit-learn
!pip install graphviz
!pip install seaborn
```

## Screen shots

<img src = "images/application UI 1.jpeg" width = 300>
<img src = "images/application UI 2.jpeg" width = 300>

## Program files:
1. <b>Data Generator.ipynb:</b> Jupyter lab file that collects images from the webcam and stores it
respective folders. 1000 images are collected for each category.
2. <b>Data Extractor.ipynb:</b> Jupyter lab file that extracts landmarks from augmented images that
are created using images from the data set collected using the previous file. The
landmarks are exported as an npy file.
3. <b>CNN Model Train.ipynb:</b> Jupyter lab file that is used to train the data set using the landmarks
collected earlier. Performance analysis is done and it exports the model for later use.
4. <b> ASL App.ipynb:</b> python file where an application is created using Tkinter that uses the model
created before to perform live translation from a webcam.


## ASL character set
<img src = "images/ASL_Alphabet.jpg" width = 300>
