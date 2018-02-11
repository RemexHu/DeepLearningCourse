import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys

class NeuralNetwork(object):
    """
    Abstraction of neural network.
    Stores parameters, activations, cached values.
    Provides necessary functions for training and prediction.
    """
    def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0):
        """
        Initializes the weights and biases for each layer
        :param layer_dimensions: (list) number of nodes in each layer
        :param drop_prob: drop probability for dropout layers. Only required in part 2 of the assignment
        :param reg_lambda: regularization parameter. Only required in part 2 of the assignment
        """
        np.random.seed(1)
        #Initiate b as a vector, needs to be modified in affineForward Function
        self.parameters = {0: {'W': np.random.randn(1024, 3072) / np.sqrt(3072),'b': np.zeros(1024)},
                           1: {'W': np.random.randn(256, 1024) / np.sqrt(1024),'b': np.zeros(256)},
                           2: {'W': np.random.randn(10, 256) / np.sqrt(256),'b': np.zeros(10)}}
        self.num_layers = 3
        self.drop_prob = 0.2
        self.reg_lambda = 0.001
        self.num_categories = 10

        # init parameters

    def affineForward(self, A, W, b):
        """
        Forward pass for the affine layer.
        :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
        the number of samples
        :returns: the affine product WA + b, along with the cache required for the backward pass
        """
        cache = {}

        batch_size = A.shape[1]

        #Modify b -> b_mod to adapt to the matrix calculation
        b_mod = np.tile(b, (batch_size, 1)).T

        Z = np.dot(W, A) + b_mod

        # Store vector b instead of b_mod
        cache.update({'W':W, 'b':b, 'A': A, 'Z': Z })

        return Z, cache



    def activationForward(self, A, activation="relu"):
        """
        Common interface to access all activation functions.
        :param A: input to the activation function
        :param prob: activation funciton to apply to A. Just "relu" for this assignment.
        :returns: activation(A)
        """
        return self.relu(A)

    def relu(self, X):

        # Note: np.max != np.maximum

        return np.maximum(X, 0)



    def dropout(self, A, prob):
        """
        :param A: Activation
        :param prob: drop prob
        :returns: tuple (A, M)
            WHERE
            A is matrix after applying dropout
            M is dropout mask, used in the backward pass
        """

        return A, M

    def softmax(self, A):
        """
        Softmax is the activation function of output layer.
        :returns: output
        """
        shiftx = A - np.max(A, axis = 0)
        exps = np.exp(shiftx)
        output = exps / np.sum(exps, axis = 0)
        return output


    def forwardPropagation(self, X):
        """
        Runs an input X through the neural network to compute activations
        for all layers. Returns the output computed at the last layer along
        with the cache required for backpropagation.
        :returns: (tuple) AL, cache
            WHERE
            AL is activation of last layer
            cache is cached values for each layer that
                     are needed in further steps
        """

        cache_all = {}
        A = X
        for layer in range(self.num_layers - 1):
            Z, cache = self.affineForward(A, self.parameters[layer]['W'], self.parameters[layer]['b'])
            U = self.activationForward(Z, activation = 'relu')
            cache_all.update( {layer: cache} )
            A = U

        output_Z, cache = self.affineForward(A, self.parameters[self.num_layers - 1]['W'], self.parameters[self.num_layers - 1]['b'])
        AL = self.softmax(output_Z)
        cache_all.update({self.num_layers - 1: cache})

        return AL, cache_all

    def costFunction(self, AL, y):
        """
        :param AL: Activation of last layer, shape (num_classes, S)
        :param y: labels, shape (S)
        :param alpha: regularization parameter
        :returns cost, dAL: A scalar denoting cost and the gradient of cost
        """
        # compute loss

        batch_size = y.shape[0]

        # get the CORRECT prediction softmax VALUE of each sample
        # AL is the matrix after softmax
        prob = AL[y, range(y.shape[0])]

        #get the prediction LABEL to calculate training acc
        label = np.argmax(AL, axis = 0)

        num_corr = np.sum(label == y)
        acc = num_corr / batch_size

        #print(prob)
        Loss = np.sum( - np.log(prob))


        y_sparse = self.sparseMatrix(y)


        reg_cost = 0

        if self.reg_lambda > 0:
        # add regularization
            for layer in range(self.num_layers):
                W = self.parameters[layer]['W']
                reg_cost += np.sum(W ** 2) / 2

        reg_coeff = 0.20
        reg_cost *= reg_coeff

        cost = Loss + reg_cost / batch_size


        # gradient of cost
        dAL = AL - y_sparse
        return cost, dAL, acc

    def sparseMatrix(self, label):
        # this function is to create a sparse matrix in order to have a matrix only at the right position to be 1, other values are all 0
        # label is a vector shape (S, )
        M = np.zeros([self.num_categories, label.shape[0]])
        M[label, np.arange(label.shape[0])] = 1

        return M


    def affineBackward(self, dA_prev, cache):
        """
        Backward pass for the affine layer.
        :param dA_prev: gradient from the next layer.
        :param cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """
        A = cache['A']
        W = cache['W']

        dW = np.dot(dA_prev, A.T)
        db = np.sum(dA_prev, axis = 1)
        dA = np.dot(W.T, dA_prev)

        return dA, dW, db

    def activationBackward(self, dA, cache, activation="relu"):
        """
        Interface to call backward on activation functions.
        In this case, it's just relu.
        """

        # dA is the output of act function (dU)
        # return the input before act function (dZ = dU * dReLu)

        return self.relu_derivative(cache['Z']) * dA


    def relu_derivative(self, Z):

        return 1. * (Z > 0)


    def dropout_backward(self, dA, cache):



        return dA

    def backPropagation(self, dAL, Y, cache):
        """
        Run backpropagation to compute gradients on all paramters in the model
        :param dAL: gradient on the last layer of the network. Returned by the cost function.
        :param Y: labels
        :param cache: cached values during forwardprop
        :returns gradients: dW and db for each weight/bias
        """
        gradients = {}
        # note that dAL has calculated act derivative, just do affine backforward

        dA, dW, db = self.affineBackward(dAL, cache[self.num_layers -1])

        # add the derivative of regulation
        dW += self.reg_lambda * cache[self.num_layers - 1]['W']
        gradients.update({self.num_layers - 1: {'W':dW, 'b':db} })

        dA_prev = dA

        for layer in range(self.num_layers - 2, -1, -1):
            dZ = self.activationBackward(dA_prev, cache[layer], activation="relu")
            dA, dW, db = self.affineBackward(dZ, cache[layer])
            gradients.update({layer: {'W': dW, 'b': db}})

            dA_prev = dA

        return gradients


    def updateParameters(self, gradients, alpha):
        """
        :param gradients: gradients for each weight/bias
        :param alpha: step size for gradient descent
        """
        # we are not updating params layer by layer, instead, we have the gradients of all layers and update them at one time

        for layer in range(self.num_layers):
            self.parameters[layer]['W'] -= alpha * gradients[layer]['W']
            self.parameters[layer]['b'] -= alpha * gradients[layer]['b']



    def train(self, X, y, iters=1000, epoch = 500, alpha=0.0001, batch_size=100, print_every=100):
        """
        :param X: input samples, each column is a sample
        :param y: labels for input samples, y.shape[0] must equal X.shape[1]
        :param iters: number of training iterations
        :param alpha: step size for gradient descent
        :param batch_size: number of samples in a minibatch
        :param print_every: no. of iterations to print debug info after
        """

        for cur_epoch in range(epoch):

            X_batch, y_batch = self.get_batch(X, y, batch_size)

            for i, pairs in enumerate(zip(X_batch,y_batch)):
                # pairs[0] is X_i, pairs[1] is y_i IN ONE BATCH

                X_i, y_i = pairs[0], pairs[1]

                AL, cache_all = self.forwardPropagation(X_i)

                cost, dAL, acc = self.costFunction(AL, y_i)

                grad = self.backPropagation(dAL, y_i, cache_all)

                self.updateParameters(grad, alpha)

                if i % print_every == 0: print('Cost = {} train acc = {}'.format(cost, acc))





    def predict(self, X):
        """
        Make predictions for each sample
        """
        pass
        return y_pred



    def get_batch(self, X, y, batch_size):
        """
        Return minibatch of samples and labels

        :param X, y: samples and corresponding labels
        :parma batch_size: minibatch size
        :returns: (tuple) X_batch, y_batch
        """

        num_example = y.shape[0]

        X_batch, y_batch = [], []

        for i in range(0, num_example, batch_size):
            X_batch_i = X[:, i : i + batch_size]
            y_batch_i = y[i: i + batch_size]

            X_batch.append(X_batch_i)
            y_batch.append(y_batch_i)


        return X_batch, y_batch






# Functions to load data, DO NOT change these

def get_labels(folder, label2id):
    """
    Returns vector of labels extracted from filenames of all files in folder
    :param folder: path to data folder
    :param label2id: mapping of text labels to numeric ids. (Eg: automobile -> 0)
    """
    files = get_files(folder)
    y = []
    for f in files:
        y.append(get_label(f, label2id))
    return np.array(y)


def one_hot(y, num_classes=10):
    """
    Converts each label index in y to vector with one_hot encoding
    One-hot encoding converts categorical labels to binary values
    """
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[y] = 1
    return y_one_hot.T


def get_label_mapping(label_file):
    """
    Returns mappings of label to index and index to label
    The input file has list of labels, each on a separate line.
    """
    with open(label_file, 'r') as f:
        id2label = f.readlines()
        id2label = [l.strip() for l in id2label]
    label2id = {}
    count = 0
    for label in id2label:
        label2id[label] = count
        count += 1
    return id2label, label2id


def get_images(folder):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    files = get_files(folder)
    images = []
    count = 0

    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count, len(files)))
        img_arr = get_img_array(f)
        img_arr = img_arr.flatten() / 255.0
        images.append(img_arr)
    X = np.column_stack(images)

    return X


def get_train_data(data_root_path):
    """
    Return X and y
    """
    train_data_path = data_root_path + 'train'
    id2label, label2id = get_label_mapping(data_root_path + 'labels.txt')
    print(label2id)
    X = get_images(train_data_path)
    y = get_labels(train_data_path, label2id)
    return X, y


def save_predictions(filename, y):


    """
    Dumps y into .npy file
    """
    np.save(filename, y)


def get_train_data(data_root_path):
    """
    Return X and y
    """
    train_data_path = data_root_path + 'train'
    id2label, label2id = get_label_mapping(data_root_path + 'labels.txt')
    print(label2id)
    X = get_images(train_data_path)
    y = get_labels(train_data_path, label2id)
    return X, y


def get_label_mapping(label_file):
    """
    Returns mappings of label to index and index to label
    The input file has list of labels, each on a separate line.
    """
    with open(label_file, 'r') as f:
        id2label = f.readlines()
        id2label = [l.strip() for l in id2label]
    label2id = {}
    count = 0
    for label in id2label:
        label2id[label] = count
        count += 1
    return id2label, label2id


def get_images(folder):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    files = get_files(folder)
    images = []
    count = 0

    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count, len(files)))
        img_arr = get_img_array(f)
        img_arr = img_arr.flatten() / 255.0
        images.append(img_arr)
    X = np.column_stack(images)

    return X


def get_files(folder):
    """
    Given path to folder, returns list of files in it
    """
    filenames = [file for file in glob.glob(folder + '*/*')]
    filenames.sort()
    return filenames


def get_img_array(path):
    """
    Given path of image, returns it's numpy array
    """
    return scipy.misc.imread(path)


def get_labels(folder, label2id):
    """
    Returns vector of labels extracted from filenames of all files in folder
    :param folder: path to data folder
    :param label2id: mapping of text labels to numeric ids. (Eg: automobile -> 0)
    """
    files = get_files(folder)
    y = []
    for f in files:
        y.append(get_label(f, label2id))
    return np.array(y)


def get_label(filepath, label2id):
    """
    Files are assumed to be labeled as: /path/to/file/999_frog.png
    Returns label for a filepath
    """
    tokens = filepath.split('/')
    label = tokens[-1].split('_')[1][:-4]
    if label in label2id:
        return label2id[label]
    else:
        sys.exit("Invalid label: " + label)


data_root_path = '/Users/runchenmac/Desktop/DEEPLEARNING/HW1/cifar10-hw1/'
X_train, y_train = get_train_data(data_root_path) # this may take a few minutes
X_test = get_images(data_root_path + 'test')
print('Data loading done')
layer_dimensions = [X_train.shape[0], 1024, 256, 10]

layer_dimensions = [X_train.shape[0], 1024, 256, 10]  # including the input and output layers
NN = NeuralNetwork(layer_dimensions)

NN.train(X_train, y_train, iters = 1000, epoch = 500, alpha = 0.0001, batch_size = 100, print_every = 100)