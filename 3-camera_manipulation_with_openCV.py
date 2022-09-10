from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet")
model.summary()
from PIL import Image
from keras import preprocessing
import time
camera = cv2.VideoCapture(0)
camera_height = 500

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect*camera_height)
    frame = cv2.resize(frame, (res, camera_height))
    cv2.rectangle(frame , (300, 75), (650, 425), (240, 100, 0), 2)
    roi = frame[75+2:425-2, 300+2:650-2]
    roi = cv2.resize(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (399, 399))
    roi = inception_v3.preprocess_input(roi)
    roi2 = np.array([cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)])
    prediction = model.predict(roi2)
    labels = inception_v3.decode_predictions(prediction, top=3)[0]
    label_1 = '{} - {}%', format(labels[0][1], int(labels[0][2]*100))
    cv2.putText(frame, label_1, (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9 (20, 20, 240), 2)
    cv2.imshow("REAL TIME OBJECT DETECTION", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
       break
camera.release()
cv2.destroyAllWindows()

