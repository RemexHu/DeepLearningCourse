from __future__ import print_function
import tensorflow as tf
import numpy as np
import cifar_utils
from cnn_model import training
import cnn_input

data_root_path = '/home/runchen/Downloads/HW1_data/cifar10-hw1/'
X_train, y_train = cnn_input.get_train_data(data_root_path)

X_train = X_train.T

#X_train, y_train = cifar_utils.load_data(mode='train')

# Data organizations:
# Train data: 49000 samples from original train set: 1~49000
# Validation data: 1000 samples from original train set: 49000~50000
num_training = 49000
num_validation = 1000

perm = np.random.permutation(49000)
X_train = X_train[perm]
y_train = y_train[perm]


X_val = X_train[-num_validation:, :]
y_val = y_train[-num_validation:]

X_train = X_train[:num_training, :]
y_train = y_train[:num_training]

mean_image = np.mean(X_train, axis=0)
X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
X_val = X_val.astype(np.float32) - mean_image

X_train /= np.std(X_train, axis=0)
X_val /= np.std(X_val, axis=0)

X_train = X_train.reshape([-1,32,32,3])
X_val = X_val.reshape([-1,32,32,3])

mask = np.random.choice(X_train.shape[0], 49000, replace=False)


X_distorted = X_train[mask, :, :, :]
y_distorted = y_train[mask]

X_distorted = cifar_utils.distorted_inputs(X_distorted, 49000)


# Preprocessing: subtract the mean value across every dimension for training data, and reshape it to be RGB size
X_train_total = np.vstack((X_train, X_distorted))
y_train_total = np.hstack((y_train, y_distorted))


print('Train data shape: ', X_train_total.shape)
print('Train labels shape: ', y_train_total.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

tf.reset_default_graph()
training(X_train_total, y_train_total, X_val, y_val,
             conv_featmap=[192, 192, 96, 96, 48, 48],
             fc_units=[512, 256],
             conv_kernel_size=[3, 3, 3, 3, 3, 3],
             pooling_size=[2, 2, 2],
             l2_norm=0.01,
             seed=235,
             learning_rate=0.0005,
             epoch=20,
             batch_size=100,
             verbose=False,
             pre_trained_model='runchennet_1519449479')


#1519279787
#1519307318

# original cifar, no batch shuffle, no data augmentation. 100 epochs: 59.7%
# original cifar, with batch shuffle, no data augmentation. 100 epoccs: 60.0%
# enhanced cifar(60k), with batch shuffle. 100 epochs: 58.8%
# enhanced cifar(70k), with batch shuffle. 100 epochs: 58.7%
# enhanced cifar(80k), with batch shuffle. 100 epochs: 59.0%

# complex CNN, with original cifar, with batch shuffle. 100 epochs: 58.6%
# complex CNN, with original cifar, with batch shuffle, with decay lr. hard to select
# complex CNN, with enhanced cifar(70k), with batch shuffle. 100 epochs:

# complex CNN, with enhanced cifar(90k), with batch shuffle. with batch_norm. extremly efficient! over 70%, but overfitting. lr=0.001

# add dropout after each fc_layer. 10 epochs 71.5% still over fitting lr=0.0008
# 80k, lr=0.0007, add dropout after conv layer 68%

# using the right training set super easy to train, lr is a little bit too high.
