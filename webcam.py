import os
import cv2
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import create_encodings
import api


def init():
    df = pd.read_csv('data/encoded-images-data.csv')
    face_encodings = df.iloc[:, 1:-1].values
    person_names = df.iloc[:, -1].values

    # Loading Cascade Classifiers
    faceClassifier = cv2.CascadeClassifier(
        'opencv/haarcascade_frontalface_default.xml')
    # faceClassifier = cv2.CascadeClassifier(
    #     'opencv/haarcascade_frontalface_alt.xml')
    # faceClassifier = cv2.CascadeClassifier('opencv/lbpcascade_frontalface.xml')
    openEyesClassifier = cv2.CascadeClassifier(
        'opencv/haarcascade_eye_tree_eyeglasses.xml')
    leftEyeClassifier = cv2.CascadeClassifier(
        'opencv/haarcascade_lefteye_2splits.xml')
    rightEyeClassifier = cv2.CascadeClassifier(
        'opencv/haarcascade_righteye_2splits.xml')

    # Loading Model
    model = api.load_eye_status_model()

    return (model, face_encodings, person_names, faceClassifier, openEyesClassifier, leftEyeClassifier, rightEyeClassifier)


def isBlinking(history, maxFrames):
    '''@history:  A string containing the history of eyes status 
                where a '0' means that the eyes were closed and '1' open.
        @maxFrames: The maximal number of successive frames where an eye is closed'''
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False


def detect_and_display(model, face_encodings, person_names, faceClassifier, openEyesClassifier, leftEyeClassifier, rightEyeClassifier, eyes_detected):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_rects = faceClassifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # for individual detected faces
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            gray_face = gray[y:y+h, x:x+w]
            
            # Encode the face into a 128-d embeddings vector
            face_encodings_in_image = api.get_face_encodings(face)
            if face_encodings_in_image:
                # find matched person
                match = api.find_match(
                    face_encodings, person_names, face_encodings_in_image[0])

                # Eyes detection
                # check first if eyes are open (with glasses taking into account)
                eyes = []
                open_eyes_glasses = openEyesClassifier.detectMultiScale(
                    gray_face,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)

                if len(open_eyes_glasses) == 2:
                    eyes_detected[match] += '1'
                    for (ex, ey, ew, eh) in open_eyes_glasses:
                        cv2.rectangle(frame, (x+ex, y+ey),
                                      (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                else:
                    # separate the face into left and right sides
                    left_face = frame[y:y+h, x+int(w/2):x+w]
                    left_face_gray = gray[y:y+h, x+int(w/2):x+w]

                    right_face = frame[y:y+h, x:x+int(w/2)]
                    right_face_gray = gray[y:y+h, x:x+int(w/2)]

                    # Detect the left eye
                    left_eye = leftEyeClassifier.detectMultiScale(
                        left_face_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE)

                    # Detect the right eye
                    right_eye = rightEyeClassifier.detectMultiScale(
                        right_face_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags = cv2.CASCADE_SCALE_IMAGE)

                    eye_status = '1'
                    IMG_SIZE = 24

                    # For each eye check wether the eye is closed.
                    # If one is closed we conclude the eyes are closed
                    for (ex,ey,ew,eh) in right_eye:
                        color = (0,255,0)
                        pred = api.predict_eye_status(cv2.resize(right_face[ey:ey+eh,ex:ex+ew], (IMG_SIZE, IMG_SIZE)),model)
                        if pred == 'closed':
                            eye_status='0'
                            color = (0,0,255)
                        cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                    
                    for (ex,ey,ew,eh) in left_eye:
                        color = (0,255,0)
                        pred = api.predict_eye_status(cv2.resize(left_face[ey:ey+eh,ex:ex+ew], (IMG_SIZE, IMG_SIZE)),model)
                        if pred == 'closed':
                            eye_status='0'
                            color = (0,0,255)
                        cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                    eyes_detected[match] += eye_status


                # Each time, we check if the person has blinked
                # If yes, we display its name
                if isBlinking(eyes_detected[match],3):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Display name
                    y = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(frame, match, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Webcam', frame)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    (model, face_encodings, person_names, faceClassifier,
     openEyesClassifier, leftEyeClassifier, rightEyeClassifier) = init()

    eyes_detected = defaultdict(str)

    detect_and_display(model, face_encodings, person_names, faceClassifier,
                       openEyesClassifier, leftEyeClassifier, rightEyeClassifier, eyes_detected)
