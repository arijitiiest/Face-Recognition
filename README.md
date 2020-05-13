# Face Recognition

Machine Learning Project to recognise faces from an image.

Build with [dlib's](http://dlib.net/) face-recognition model

# model used
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [opencv classifier](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

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
- Thanks to Shehzad Noor Taus Priyo for this [blog](https://towardsdatascience.com/facial-recognition-using-deep-learning-a74e9059a150) post
