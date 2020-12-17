import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
def load_cv2_data(num, root_path):
    img_path = 'img/'
    X = np.zeros((num,50,50,3))   
    for file in range(num):
        image = cv2.imread(root_path + img_path + '{}'.format(file) +'.png', 1)
        X[file] = cv2.resize(image,(50,50))    # resize or compress the images
    return X

# Load the csv
def load_label(label,root_path):
    labels = pd.read_csv(root_path + 'labels.csv', delim_whitespace = True, header=0)
    Y = labels[[label]]
    Y = np.array(Y)
    return Y

# The one used in main.py
def pre_processing(num, label, root_path, split = True):
    X = load_cv2_data(num, root_path)
    Y = load_label(label, root_path)    
    # determine whether to split the data into training set and test set
    if (split == True):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
        return x_train, x_test, y_train, y_test
    else:
        return X, Y