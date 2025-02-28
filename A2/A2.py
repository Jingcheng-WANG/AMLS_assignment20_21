import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings("ignore")

# SVM Hyper-parameter Tuning
def A2_SVM_ParameterTuning(x_train, y_train):
    param_grid = {'kernel':('linear','poly','rbf'), 'C':[1, 10]}
    grid = GridSearchCV(SVC(),param_grid = param_grid, cv = 4)   # GridSearchCV
    grid.fit(x_train, y_train)
    return grid.best_params_

# SVM modeling
def A2_SVM(x_train,y_train):
    model = SVC(kernel='poly', degree=3, C = 10)   # Build the model
    model.fit(x_train,y_train)   # Fit the model
    return model

# Accuracy by useing SVM
def A2_acc(x_test, y_test, model):
    y_pred = model.predict(x_test)   # Predict the results
    acc = accuracy_score(y_test,y_pred)   # print the accuracy
    return acc

# ======================================================================================================================
# ===============================================Use only when Tuning===================================================、

# RF Hyper-parameter Tuning
def A2_RF_ParameterTuning(x_train, y_train):
    param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((100,150,200))}
    grid = GridSearchCV(RandomForestClassifier(),param_grid = param_grid, cv = 4)   # GridSearchCV
    grid.fit(x_train, y_train)
    return grid.best_params_

# RF modeling
def A2_RF(x_train,y_train):
    model = RandomForestClassifier(min_samples_split = 6, n_estimators = 200, max_depth = 5)   # Build the model
    model.fit(x_train, y_train)   # Fit the model
    return model

# Learning curve
def plot_learning_curve(estimator, title, x_train, y_train, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
 
    plt.legend(loc="best")
    return plt

#title = r"Learning Curves (Task A2, SVM, poly kernel, C=10)"
#cv = 4
#plot_learning_curve(model, title, x_train, y_train, (0.7, 1.01), cv=cv, n_jobs=1)