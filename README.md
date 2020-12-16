 AMLS_assignment20_21-SN20040326
 ======
 ## Background
The development of biometric recognition technology has greatly promoted modern security, administration and business systems. Facial recognition, the most important part of biometric recognition, is defined as a technology that can recognize people based on their facial features. It is based on sophisticated mathematical AI and machine learning algorithms that can capture, store and analyze facial features to match individual images in pre-existing databases. The aim of this project is to further develop programming skill and comprehensive understanding of machine learning systems studied during this course. In this project, we downloaded the face data set which included [5000 celebrities and 10000 face of anime characters](https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view). Four tasks are included in this project:<br>
* A1 Gender detection: male or female.<br>
* A2 Emotion detection: smiling or not smiling.<br>
* B1 Face shape recognition: 5 types of face shapes.<br>
* B2 Eye color recognition: 5 types of eye colors.<br>
In task A, traditional machine learning methods SVM and RF are adopted. In task B, we adopted and compared traditional machine learning methods, RF with MLP and CNN. The code can be found through the link at the bottom of this page .
## Install
### Requirement
* Python 3.3+<br>
* macOS or Linux (Windows
### Installation Options
Go check them out if you do not have them locally installed.
```python
import numpy
import pandas
import sklearn
import cv2
import dlib
import matplotlib
import tensorflow
```
Make sure your tensorflow version is higher than 2.0 <br>
```python
print(tf.__version__)
```
## File Description
### Base Directory
* **main.py** can be excuted to run the whole project.
* **shape_predictor_68_face_landmarks.dat** is a face recognition database used for detecting 68 feature points.
* The file folder A1, A2, B1, B2 contain code fles for each task. Basically, they cannot run separately but need to rely on the support of main.py
* **Dlib_load_data.py** is done independently by us for importing 68 feature points of images and labels.
* **CV2_load_data.py** is done independently by us for importing image pixel matrix sets and labels.

### Folder A1
### Folder A2
### Folder B1
### Folder B2


## Contributors
This project exists thanks to all the people who contribute.<br>
[![Jingcheng Wang](https://avatars3.githubusercontent.com/u/72794136?s=60&v=4 "Jingcheng Wang")](https://github.com/Jingcheng-WANG)
