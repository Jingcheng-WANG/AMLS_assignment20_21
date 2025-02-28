 AMLS_assignment20_21-SN20040326
 ======
 ***IMPORTANT: If you want to mark this code, please use [Google Drive](https://drive.google.com/file/d/1A3-sWz8V8xHOasjV4R9PtThAhum3yO9K/view?usp=sharing). The reason is that this code needs to place shape_predictor_68_face_landmarks.dat and main.py in the same directory, but this file is larger than 25 MB so that it cannot be uploaded to GitHub***
 
 ## Background
The development of biometric recognition technology has greatly promoted modern security, administration and business systems. Facial recognition, the most important part of biometric recognition, is defined as a technology that can recognize people based on their facial features. It is based on sophisticated mathematical AI and machine learning algorithms that can capture, store and analyze facial features to match individual images in pre-existing databases. The aim of this project is to further develop programming skill and comprehensive understanding of machine learning systems studied during this course. In this project, we downloaded the face data set which included [5000 celebrities and 10000 face of anime characters](https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view), and the test code can be downloaded [here](https://drive.google.com/file/d/1Yt4C0p86-yySY45QwsfWMUlfnd9plQWx/view). Four tasks are included in this project:<br>
* A1 Gender detection: male or female.<br>
* A2 Emotion detection: smiling or not smiling.<br>
* B1 Face shape recognition: 5 types of face shapes.<br>
* B2 Eye color recognition: 5 types of eye colors.<br>

In task A, traditional machine learning methods SVM and RF are adopted. In task B, we adopted and compared traditional machine learning methods (RF) with Neural Networks (MLP and CNN).
## Install
### Requirement
* Python 3.3+<br>
* macOS, Linux or Windows
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
* **Datasets folder**: Please use your own dataset.
* The file folder A1, A2, B1, B2 contain code fles for each task. Basically, they cannot run separately but need to rely on the support of main.py
* **Dlib_load_data.py** is done independently by us for importing 68 feature points of images and labels.<br>
>>In this file, we shall only look at and call
>>```python
>>def pre_processing(num, label, root_path, point, split = True)
>>```
>>*num* is the number of images to be imported, *label* is the index of the labels, *root_path* is the location path of the filea called, *point* is the number of feature points needed, *split* determines whether split the dataset into training set and validation set.
* **CV2_load_data.py** is done independently by us for importing image pixel matrix sets and labels.
>>In this file, we shall only look at and call
>>```python
>>def pre_processing(num, label, root_path, split = True)
>>```
>>*num* is the number of images to be imported, *label* is the index of the labels, *root_path* is the location path of the filea called, *split* determines whether split the dataset into training set and validation set. The imported pixels will be compressed to 50*50. If you are not satisfied with this compression,
>>```python
>>def load_cv2_data(num, root_path):
>>     img_path = 'img/'
>>     X = np.zeros((num,50,50,3))
>>     for file in range(num):
>>         image = cv2.imread(root_path + img_path + '{}'.format(file) +'.png', 1)
>>         X = cv2.resize(image,(50,50))
>>    return X
>>```
>>you can modify line 3 and line 6
### Folder A1
* **A1.py** contains a lot of defined function which can be called in **main.py**. 
>>Specifically, it includes hyper-parameter selection by using GridSearchCV `A1_SVM_ParameterTuning()` `A1_RF_ParameterTuning()`, model construction `A1_SVM()` `A1_RF()`, accuracy report `A1_acc()`, and learning curve plotting `plot_learning_curve()`
### Folder A2
* **A2.py** contains a lot of defined function which can be called in **main.py**. 
>>Specifically, it includes hyper-parameter selection by using GridSearchCV `A2_SVM_ParameterTuning()` `A2_RF_ParameterTuning()`, model construction `A2_SVM()` `A2_RF()`, accuracy report `A2_acc()`, and learning curve plotting `plot_learning_curve()`
### Folder B1
* **B1 B1_CNN.py** is the most accurate model and will be called in **main.py**. 
>>Specifically, it includes model construction `B1_CNN()`, accuracy report `B1_acc()`, confusion matrix plotting `plot_confusion_matrix()`, loss curve plotting `plot_loss_curve()`, accuracy curve plotting `plot_accuracy_curve()`.
* **B1_MLP.py** is not called in **main.py** and can be executed separately. 
>>Specifically, it includes On hot coding transformation `On_Hot_Coding()`, model construction `allocate_weights_and_biases()+multilayer_perceptron()`, hyper-parameter setting, accuracy report<br>
>>To ensure that it can be run separately, please `import Dlib_load_data` in the correct path and copy **shape_predictor_68_face_landmarks.dat**
* **B1_RF.py** is not called in **main.py** and cannot be executed separately.
>>Specifically, it includes hyper-parameter selection by using GridSearchCV `B1_RF_ParameterTuning()`, model construction `B1_RF()`, accuracy report `B1_acc()`, and learning curve plotting `plot_learning_curve()`
### Folder B2
* **B2 B2_CNN.py** will be called in **main.py**. 
>>Specifically, it includes model construction `B2_CNN()`, accuracy report `B2_acc()`, confusion matrix plotting `plot_confusion_matrix()`, loss curve plotting `plot_loss_curve()`, accuracy curve plotting `plot_accuracy_curve()`.

## Usage
Make sure your own datasets are in the same directory with **main.py** and have the following structure (subfiles)<br>
* Datasets
> * cartoon_set
> * cartoon_set_test
> * celeba
> * celeba_test<br>

Remember, if datasets are placed in the wrong path or missing subfiles, the main fuction will not run. If you really cannot unify datasets due to some specific reasons, then you are recommended to read the description of `Dlib_load_data.pre_processing()` and `CV2_load_data.pre_processing()` in the file description and try to modify the imported parameters in **main.py**.<br>
<br>
Next, just enter the following code in the command window
```
python main.py
```
Then, you can see the accuracy socre for training set and test set in each task<br>

***Hint: It will takes few minutes to executed.*** (The testing computer is configured as CPU: i7-10750H, RAM: 16GB, GPU: Nvidia GeForce RTX 2060, and it taks about 8 min to run this code)
## Contributors
This project exists thanks to all the people who contribute.<br>
[![Jingcheng Wang](https://avatars3.githubusercontent.com/u/72794136?s=60&v=4 "Jingcheng Wang")](https://github.com/Jingcheng-WANG)
