import dlib
import numpy as np


# Globals
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

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
        return person_names[min_index]+" ({0:.2f})".format(min_value)
    if min_value < 0.65:
        return person_names[min_index]+"?"+" ({0:.2f})".format(min_value)
    return 'Not Found'
    