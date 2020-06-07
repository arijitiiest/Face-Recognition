import os
import cv2
import pickle
import numpy as np
import pandas as pd
import api


def load_images(faces_folder_path):
    ''' 1. Load training images
        2. Filter the image files
        3. Add person name from filename
        
        Returns: Images path for each persons'''
    
    image_filenames = filter(lambda x : (x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')), os.listdir(faces_folder_path))
    image_filenames = sorted(image_filenames)
    person_names = [x[:-4] for x in image_filenames]
    
    full_path_to_images = [faces_folder_path + '/' + x for x in image_filenames]
    return full_path_to_images, person_names


def create_face_encodings(faces_folder_path):
    # Encode the face into a 128-d embeddings vector

    full_path_to_image, person_names = load_images(faces_folder_path)
    face_encodings = []

    for path_to_image in full_path_to_image:
        face = cv2.imread(path_to_image)

        encodings = api.get_face_encodings(face)

        if len(encodings) != 1:
            print('Warning: Images should have one and only face per image: ' + path_to_image + " It has " + str(len(encodings)))
            exit()

        face_encoding = encodings[0]
        face_encodings.append(face_encoding)
    
    return face_encodings, person_names

if __name__ == "__main__":
    faces_folder_path = 'data/training_images'
    encoding_file_path = 'data/encoded-images-data.csv'

    face_encodings, person_names = create_face_encodings(faces_folder_path)

    # Save encodings to csv file
    df1 = pd.DataFrame(face_encodings)
    df2 = pd.DataFrame(person_names)

    df = pd.concat([df1, df2], axis=1)

    # if file with same name already exists, backup the old file
    if os.path.isfile(encoding_file_path):
        print("{} already exists. Backing up.".format(encoding_file_path))
        os.rename(encoding_file_path, "{}.bak".format(encoding_file_path))

    df.to_csv(encoding_file_path)
