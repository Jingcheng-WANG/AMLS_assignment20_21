# import machine learning models
import A1.A1 as A1
import A2.A2 as A2
import B1.B1_CNN as B1
import B2.B2_CNN as B2

# import feature extraction
import Dlib_load_data
import CV2_load_data

print("The library is loaded successfully and the data is being imported......")
# ======================================================================================================================
# ====================================================Data preprocessing================================================
# Task A1
x_train_A1, x_val_A1, y_train_A1, y_val_A1 = Dlib_load_data.pre_processing(5000, 'gender', './Datasets/celeba/', 68, split = True)
x_test_A1, y_test_A1 = Dlib_load_data.pre_processing(1000, 'gender', './Datasets/celeba_test/', 68, split = False)
# Task A2
x_train_A2, x_val_A2, y_train_A2, y_val_A2 = Dlib_load_data.pre_processing(5000, 'smiling', './Datasets/celeba/', 68, split = True)
x_test_A2, y_test_A2 = Dlib_load_data.pre_processing(1000, 'smiling', './Datasets/celeba_test/', 68, split = False)
# Task B1
x_train_B1, x_val_B1, y_train_B1, y_val_B1 = CV2_load_data.pre_processing(10000, 'face_shape', './Datasets/cartoon_set/', split = True)
x_test_B1, y_test_B1 = CV2_load_data.pre_processing(2500, 'face_shape', './Datasets/cartoon_set_test/', split = False)
# Task B2
x_train_B2, x_val_B2, y_train_B2, y_val_B2 = CV2_load_data.pre_processing(10000, 'eye_color', './Datasets/cartoon_set/', split = True)
x_test_B2, y_test_B2 = CV2_load_data.pre_processing(2500, 'eye_color', './Datasets/cartoon_set_test/', split = False)
# ======================================================================================================================
#====================================================Modeling===========================================================
# Task A1
model_A1 = A1.A1_SVM(x_train_A1,y_train_A1)                 # Build model object.
acc_A1_train = A1.A1_acc(x_train_A1, y_train_A1, model_A1) # Train model based on the training set
acc_A1_test = A1.A1_acc(x_test_A1, y_test_A1, model_A1)   # Test model based on the test set.
# Task A2
model_A2 = A2.A2_SVM(x_train_A2,y_train_A2)                 # Build model object.
acc_A2_train = A2.A2_acc(x_train_A2, y_train_A2, model_A2) # Train model based on the training set
acc_A2_test = A2.A2_acc(x_test_A2, y_test_A2, model_A2)   # Test model based on the test set.
# Task B1
model_B1 = B1.B1_CNN(x_train_B1, x_val_B1, y_train_B1, y_val_B1)  # Build model object.
acc_B1_train = B1.B1_acc(x_train_B1, y_train_B1, model_B1) # Train model based on the training set
acc_B1_test = B1.B1_acc(x_test_B1, y_test_B1, model_B1)   # Test model based on the test set.
# Task B2
model_B2 = B2.B2_CNN(x_train_B2, x_val_B2, y_train_B2, y_val_B2)  # Build model object.
acc_B2_train = B2.B2_acc(x_train_B2, y_train_B2, model_B2) # Train model based on the training set
acc_B2_test = B2.B2_acc(x_test_B2, y_test_B2, model_B2)   # Test model based on the test set.
# ======================================================================================================================
## Print out results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                       acc_A2_train, acc_A2_test,
                                                       acc_B1_train, acc_B1_test,
                                                       acc_B2_train, acc_B2_test))
