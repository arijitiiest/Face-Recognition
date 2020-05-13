import os
import cv2
import pickle
import numpy as np
import pandas as pd
import create_encodings
import api

df = pd.read_csv('data/encoded-images-data.csv')
face_encodings = df.iloc[:, 1:-1].values
person_names = df.iloc[:, -1].values

faceClassifier = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = faceClassifier.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_encodings_in_image = api.get_face_encodings(face)
        if face_encodings_in_image:
            match = api.find_match(face_encodings, person_names, face_encodings_in_image[0])
            cv2.putText(frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Webcam' ,frame)

    k = cv2.waitKey(0) & 0xff
    if k == ord('q') or k == 27:
        break


cap.release()
cv2.destroyAllWindows()
