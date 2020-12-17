import numpy as np
import pandas as pd
import Dlib_load_data      # You need to find the right path!!!!!!
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf      #If your tensorflow < 2.0, ignore

# tensorflow version < 2 is adopted
tf.disable_v2_behavior()

# Load the data and labels
x_train_B1, x_val_B1, y_train_B1, y_val_B1 = Dlib_load_data.pre_processing(10000, 'face_shape', '../Datasets/cartoon_set/', 17, split = True)
x_test_B1, y_test_B1 = Dlib_load_data.pre_processing(2500, 'face_shape', '../Datasets/cartoon_set_test/', 17, split = False)

# On hot coding transformation
def On_Hot_Coding(y_train):
    yy_train = np.zeros([y_train.shape[0],5])
    for n in range(y_train.shape[0]):
        if (y_train[n] == 0):yy_train[n] = [1,0,0,0,0]
        if (y_train[n] == 1):yy_train[n] = [0,1,0,0,0]
        if (y_train[n] == 2):yy_train[n] = [0,0,1,0,0]
        if (y_train[n] == 3):yy_train[n] = [0,0,0,1,0]
        if (y_train[n] == 4):yy_train[n] = [0,0,0,0,1]
    return yy_train

# # Match model input
x_train_B1 = x_train_B1.reshape(x_train_B1.shape[0],17,2)
x_test_B1 = x_test_B1.reshape(x_test_B1.shape[0],17,2)
x_val_B1 = x_val_B1.reshape(x_val_B1.shape[0],17,2)
y_train_B1 = On_Hot_Coding(y_train_B1)
y_val_B1 = On_Hot_Coding(y_val_B1)
y_test_B1 = On_Hot_Coding(y_test_B1)

def allocate_weights_and_biases():

    # define number of hidden layers ..
    n_hidden_1 = 1024  # 1st layer number of  neurons
    n_hidden_2 = 1024  # 2nd layer number of neurons
    n_hidden_3 = 1024   # 3nd layer number of neurons
   
    # inputs placeholders
    X = tf.placeholder("float", [None, 17, 2])
    Y = tf.placeholder("float", [None, 5])  # 2 output classes
    
    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    #images_flat = tf.contrib.layers.flatten(X)
    images_flat = tf.layers.flatten(X)  
    
    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01
    
    # define placeholders for weights and biases in the graph
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([17 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'hidden_layer3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, 5], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'bias_layer3': tf.Variable(tf.random_normal([n_hidden_3], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([5], stddev=stddev))
    }
    
    return weights, biases, X, Y, images_flat

def multilayer_perceptron():
        
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()
    
    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Hidden fully connected layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['hidden_layer3']), biases['bias_layer3'])
    layer_3 = tf.nn.relu(layer_3)
    
    # Output fully connected layer
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer, X, Y

# learning parameters
learning_rate = 1e-4
training_epochs = 1000

# display training accuracy every ..
display_accuracy_step = 30

logits, X, Y = multilayer_perceptron()

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# define training graph operation
train_op = optimizer.minimize(loss_op)

# graph operation to initialize all variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # run graph weights/biases initialization op
    sess.run(init_op)
    # begin training loop ..
    for epoch in range(training_epochs):
        # complete code below
        # run optimization operation (backprop) and cost operation (to get loss value)
        _, cost = sess.run([train_op, loss_op], feed_dict={X: x_train_B1,
                                          Y: y_train_B1})

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))
                
        if epoch % display_accuracy_step == 0:
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

            # calculate training accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy: {:.3f}".format(accuracy.eval({X: x_train_B1, Y: y_train_B1})))

    print("Optimization Finished!")

    # -- Define and run test operation -- #
        
    # apply softmax to output logits
    pred = tf.nn.softmax(logits)
        
    #  derive inffered calasses as the class with the top value in the output density function
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
    # complete code below
    # run test accuracy operation ..
    print("Train Accuracy:", accuracy.eval({X: x_train_B1, Y:y_train_B1}))
    print("Validation Accuracy:", accuracy.eval({X: x_val_B1, Y:y_val_B1}))
    
    print("Test Accuracy:", accuracy.eval({X: x_test_B1, Y:y_test_B1}))