import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D,Activation,Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

model = tf.keras.models.load_model('modelvgg3.h5')
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

def Detection():

    mapping = {"paper": 0, "rock": 1, "scissors": 2}
    reverse_mapping = {0: "paper", 1: "rock", 2: "scissors"}

    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        # Get image frame
        success, img = cap.read()
        image_pred = img
        image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
        image_pred = cv2.resize(image_pred, (200, 200), interpolation=cv2.INTER_AREA)
        image_pred = np.array(image_pred)
        image_pred = tf.keras.applications.vgg16.preprocess_input(image_pred)
        pred = model.predict(np.reshape(image_pred, (-1, 200, 200, 3)))

        # Find the hand and its landmarks
        hands, img = detector.findHands(img)

        org = (00, 100)
        fontScale = 3
        color = (0, 0, 0)
        thickness = 2

        pred_txt = reverse_mapping[np.argmax(pred)]

        if hands:
            img = cv2.putText(img, pred_txt, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                  color, thickness, cv2.LINE_AA, False)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()



Detection()




