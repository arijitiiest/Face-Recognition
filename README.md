# Face Recognition

Machine Learning Project to recognise faces with Real time face liveliness detection.

Build with [dlib's](http://dlib.net/) face-recognition model

![face](https://user-images.githubusercontent.com/53527166/84349489-0c3e1380-abd5-11ea-98ef-f3d6d5a71b54.png)


# model used
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [opencv classifier haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
- [opencv haarcascade_eye_tree_eyeglasses.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml)
- [opencv haarcascade_lefteye_2splits.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_lefteye_2splits.xml)
- [opencv haarcascade_righteye_2splits.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_righteye_2splits.xml)

# Instruction
- Clone this repository `https://github.com/arijitiiest/Face-Recognition`
- Download the models and copy to `models` folder
- Download opencv classifier and copy to `opencv` folder
  ``` bash 
  $ pip install requirements.txt
  $ mkdir data/training_images
  ```
- create images, image name should be person name
  ``` bash 
  $ python create_encodings.py 
  $ python webcam.py
  ```

# Credits
- Thanks to [Davis King](https://github.com/davisking) for creating dlib and for providing the trained facial feature
  detection and face encoding models used in this project.
- Thanks to [Adam Geitgey](https://github.com/ageitgey) whose [blog](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) inspired me to make this project.
- Thanks to [Jordon Van Eetveldt](https://github.com/Guarouba) for this [blog](https://towardsdatascience.com/real-time-face-liveness-detection-with-python-keras-and-opencv-c35dc70dafd3) post for the idea to detect liveliness of face.
