import dlib
import numpy as np
import cv2
from keras.models import model_from_json


# Globals
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
IMG_SIZE = 24   


# Compute face encodings for a face
def get_face_encodings(face):
    bounds = face_detector(face, 1)
    face_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in face_landmarks]


# Compute face differences
def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces-face, axis=1)


# Compute face matches
def find_match(known_faces, person_names, face):
    matches = get_face_matches(known_faces, face)
    min_index = matches.argmin()
    min_value = matches[min_index]
    if min_value < 0.55:
        return person_names[min_index] + "! ({0:.2f})".format(min_value)
    if min_value < 0.58:
        return person_names[min_index] + " ({0:.2f})".format(min_value)
    if min_value < 0.65:
        return person_names[min_index] + "?" + " ({0:.2f})".format(min_value)
    return 'Not Found'


# Load eye status predictor model
def load_eye_status_model():
    json_file = open('models/eye_status_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/eye_status_model.h5")    # load weights
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


# Predict eye status
def predict_eye_status(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img2 = np.reshape(np.array(img), (1, IMG_SIZE, IMG_SIZE, 1))
    prediction = model.predict(img2)
    if prediction[0][0] < 0.1:
        pred = 'closed'
    elif prediction > 0.9:
        pred = 'open'
    else:
        pred = "idk"
    # print(pred, prediction)
    return pred