import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dlib_data(num, root_path, point):
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # read the image
    img_path = 'img/'
    X = np.ones((num,point,2))
    for file in range(num):
        image = cv2.imread(root_path + img_path + '{}'.format(file) +'.jpg', 0) 
        # Use detector to find landmarks
        faces = detector(image)
        for face in faces:
            # Create landmark object
            landmarks = predictor(image=image, box=face)
            # Loop through all the points
            for n in range(0, point):
                X[file][n] = [landmarks.part(n).x,landmarks.part(n).y]
    return X

def load_label(label,root_path):
    # Load the csv
    labels = pd.read_csv(root_path + 'labels.csv', delim_whitespace = True, header=0)
    Y = labels[[label]]
    Y = np.array(Y)
    return Y

def pre_processing(num, label, root_path, point, split = True):
    X = load_dlib_data(num, root_path, point)
    Y = load_label(label, root_path)
    X = X.reshape(X.shape[0],-1)    # Match model input
    if (split == True):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
        return x_train, x_test, y_train, y_test
    else:
        return X, Y