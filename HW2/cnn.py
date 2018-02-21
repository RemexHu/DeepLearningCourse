from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar_utils import load_data
from matplotlib import pyplot as plt
from cnn_model import training



X_train, y_train = load_data(mode='train')

# Data organizations:
# Train data: 49000 samples from original train set: 1~49000
# Validation data: 1000 samples from original train set: 49000~50000
num_training = 49000
num_validation = 1000

X_val = X_train[-num_validation:, :]
y_val = y_train[-num_validation:]

X_train = X_train[:num_training, :]
y_train = y_train[:num_training]

# Preprocessing: subtract the mean value across every dimension for training data, and reshape it to be RGB size
mean_image = np.mean(X_train, axis=0)
X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
X_val = X_val.astype(np.float32) - mean_image

X_train = X_train.reshape([-1,32,32,3])/255
X_val = X_val.reshape([-1,32,32,3])/255

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)


tf.reset_default_graph()
training(X_train, y_train, X_val, y_val,
             conv_featmap=[32, 32, 64],
             fc_units=[512],
             conv_kernel_size=[5, 5, 3],
             pooling_size=[2, 2, 2],
             l2_norm=0.001,
             seed=235,
             learning_rate=1e-2,
             epoch=2000,
             batch_size=128,
             verbose=False,
             pre_trained_model=True)