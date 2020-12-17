import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf  
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# modeling
def B1_CNN(x_train, x_test, y_train, y_test):
    # add layers
    i = Input(shape=x_train[0].shape)   # input layer
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)   # convolutional layer
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)   # convolutional layer
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)   # convolutional layer
    x = Flatten()(x)   # flatten layer
    x = Dropout(0.2)(x)   # dropout layer
    x = Dense(512, activation='relu')(x)   # dense layer
    x = Dropout(0.2)(x))   # dropout layer
    x = Dense(5, activation='softmax')(x)# dense layer
    
    # Built model
    CNN = Model(i, x)

    # Compile
    CNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    # Fit
    CNN.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    
    return CNN

# accuracy score
def B1_acc(x_test, y_test, model):
    y_pred = model.predict(x_test)
    
    # Transform on-hot coding to class
    y_pred_trans = np.zeros([y_pred.shape[0],1])
    for n in range(y_pred.shape[0]):
        if (y_pred[n][0] >= 0.5):y_pred_trans[n] = 0
        if (y_pred[n][1] >= 0.5):y_pred_trans[n] = 1
        if (y_pred[n][2] >= 0.5):y_pred_trans[n] = 2
        if (y_pred[n][3] >= 0.5):y_pred_trans[n] = 3
        if (y_pred[n][4] >= 0.5):y_pred_trans[n] = 4
    acc = accuracy_score(y_test,y_pred_trans)
    return acc

# ======================================================================================================================
# ===============================================Use only when Tuning===================================================

# confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return plt

# loss curve
def plot_loss_curve(r):
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    return r

# accuracy curve
def plot_accuracy_curve(r):
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    return plt