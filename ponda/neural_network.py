from optimizer import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def linear(x):
    return x

def d_linear(x):
    x[:] = 1

def relu(x):
    return np.maximum(x, 0.01)

def d_relu(x):
    x[x > 0] = 1
    x[x <= 0] = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    np.exp(x, x)
    x[:] = x / np.power(x + 1, 2)

def tanh(x):
    ex = np.exp(x)
    nex = np.exp(-x)
    return (ex-nex)/(ex+nex)

def d_tanh(x):
    np.power(np.exp(x) + np.exp(-x), 2, x)
    x[:] = 4/x

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex, axis=0)

def d_softmax(x):
    x[:] = x/(1-x)

def mse(y, p):
    return np.power(y-p, 2)/2

def d_mse(y, p):
    return p - y

# to timestamp format hh:mm:ss
def to_timestamp(timedelta):
    seconds = int(timedelta.total_seconds())
    hours = seconds // 3600
    seconds -= hours*3600
    minutes = seconds//60
    seconds -= minutes * 60
    return '{:02}:{:02}:{:02}'.format(hours, minutes, seconds)

class Layer:
    def __init__(self, units, inputs=None, activation='linear', dropout=None):
        self.units = units
        self.inputs = inputs
        self.outputLayer = False
        self.activation = activation
        self.dropout = dropout

        if activation == 'linear':
            self.active, self.d_active = linear, d_linear

        elif activation == 'relu':
            self.active, self.d_active = relu, d_relu

        elif activation == 'sigmoid':
            self.active, self.d_active = sigmoid, d_sigmoid

        elif activation == 'tanh':
            self.active, self.d_active = tanh, d_tanh

        elif activation == 'softmax':
            self.active, self.d_active = softmax, d_softmax

        else:
            raise Exception('Unknown activation function: "{}"'.format(activation))


    # Allocating needed memory
    def alloc(self, optimizer):
        self.W = np.random.rand(self.units, self.inputs)*2-1
        self.B = np.random.rand(self.units, 1)*2-1

        if self.dropout is not None:
            self.drop_W = np.random.rand(self.units*self.inputs)
            self.drop_B = np.random.rand(self.units)
            self.drop_W[self.drop_W >= self.dropout] = 1
            self.drop_W[self.drop_W < self.dropout] = 0
            self.drop_B[self.drop_B >= self.dropout] = 1
            self.drop_B[self.drop_B < self.dropout] = 0

        if optimizer == 'adam':
            self.W_optimizer, self.B_optimizer = Adam(), Adam()
        elif optimizer == 'adadelta':
            self.W_optimizer, self.B_optimizer = Adadelta(), Adadelta()
        else:
            raise  Exception('Unknown optimizer: "{}"'.format(optimizer))

    # Use for training
    def forward(self, x):
        x.shape = (self.inputs, 1)
        self.pre_F = x
        self.O = self.W.dot(x) + self.B
        self.F = self.active(self.O)
        return self.F

    # Use for predict, x <= x.T
    def fast_forward(self, x):
        return self.active(self.W.dot(x) + self.B)

    # Back propagation
    def update(self, pre_value):
        self.d_active(self.O)
        cache = pre_value * self.O
        self.pre_value = self.W.T.dot(cache)

        if self.dropout is not None:
            # Dropout update
            np.random.shuffle(self.drop_W)
            np.random.shuffle(self.drop_B)
            index_W = self.drop_W.reshape(self.units, self.inputs) == 1
            index_B = self.drop_B.reshape(self.units, 1) == 1
            self.W[index_W] = self.W_optimizer.get(self.W, cache.dot(self.pre_F.T))[index_W]
            self.B[index_B] = self.B_optimizer.get(self.B, cache)[index_B]

        else:
            self.W = self.W_optimizer.get(self.W, cache.dot(self.pre_F.T))
            self.B = self.B_optimizer.get(self.B, cache)
            # self.W -= cache.dot(self.pre_F.T) * 1e-5
            # self.B -= cache * 1e-5

        return self.pre_value



class Model:
    available_metrics = {'acc', 'accuracy'}

    def __init__(self):
        self.layers = []
        self.hasAdd = False
        self.hasCompile = False
        self.hasFit = False

    # Add a layer
    def add(self, layer):
        if len(self.layers) > 0:
            layer.inputs = self.layers[-1].units
        self.layers.append(layer)
        self.hasAdd = True

    # Compile before "fit"
    def compile(self, optimizer='adadelta', loss='mse', dropout=None, metrics=list()):
        if not self.hasAdd:
            raise Exception('Must "add" before compile')

        # Check for loss
        if loss == 'mse':
            self.loss , self.d_loss = mse, d_mse
        else:
            raise Exception('Unknown loss function: "{}"'.format(loss))

        # Check for metric
        self.metrics = metrics
        for metric in self.metrics:
            if metric not in Model.available_metrics:
                raise Exception('Unknown metric: "{}"'.format(metric))

        for layer in self.layers:
            layer.dropout = dropout
            layer.alloc(optimizer)
        self.inputs = self.layers[0].inputs
        self.outputs = self.layers[-1].units
        self.hasCompile = True

    # predict multiple inputs
    def predict(self, x):
        if not self.hasCompile:
            raise Exception('Must "compile" before predict')

        # Check input dimension is right
        self._check_dim(x)

        x = x.T
        for i in range(0, len(self.layers)):
            x = self.layers[i].fast_forward(x)
        return x.T

    # predict multiple inputs and transfer to class
    def predict_class(self, x):
        if not self.hasCompile:
            raise Exception('Must "compile" before predict_class')

        return np.argmax(self.predict(x), axis=1)

    # test the prediction
    def test(self, x, y, display=True):
        if not self.hasCompile:
            raise Exception('Must "compile" before test')

        # Check input and output dimension is right
        self._check_dim(x, y)

        p = self.predict(x)
        test_report = {}
        message = ''

        loss = np.mean(self.loss(y, p))
        test_report['loss'] = loss
        message += 'TEST loss {:.3e}'.format(loss)

        for metric in self.metrics:
            if metric in ['acc', 'accuracy']:
                p_label = np.argmax(p, 1)
                y_label = np.argmax(y, 1)
                acc = np.sum(y_label == p_label)/len(y_label)
                test_report['acc'] = acc
                test_report['confusion_matrix'] = confusion_matrix(y_label, p_label)
                message += ' - acc {:.2f}%'.format(acc*100)

        if display:
            print(message)

        return test_report


    # Run to start training
    def fit(self, x, y, epochs, verbose=1):
        if not self.hasCompile:
            raise Exception('Must "compile" before fit')

        self.verbose = verbose
        self.epochs = epochs

        # Check input and output dimension is right
        self._check_dim(x, y)

        # Store predict value for calculating loss later
        p_tmp = np.zeros(y.shape)
        self._log_init()

        train_index = np.arange(len(x))
        for t in range(epochs):
            # Auto shuffle training order
            np.random.shuffle(train_index)
            for i in train_index:
                # Forward
                p = x[i]
                for layer in self.layers:
                    p = layer.forward(p)
                p_tmp[i] = p.reshape(-1)

                # Update, [-1::-1] => reverse list
                pre_value = self.d_loss(y[i].reshape(-1, 1), p)
                for layer in self.layers[-1::-1]:
                    pre_value = layer.update(pre_value)
            self._log(t+1, y, p_tmp)

        if self.verbose == 1:
            print()

        self.hasFit = True
        return self.history


    def plot_history(self, file_path=None):
        if not self.hasFit:
            raise Exception('Must "fit" before plot_history')

        chart_count = 1 + len(self.metrics)
        plt.figure()
        plt.subplot(1, chart_count, 1)
        plt.title('Loss')
        plt.plot(self.history['loss'])
        plt.axhline(0, color='r', linestyle='--')

        for metric in self.metrics:
            if metric in ['acc', 'accuracy']:
                plt.subplot(1, chart_count, 2)
                plt.title('Acc')
                plt.plot(self.history['acc'])
                plt.axhline(0, color='r', linestyle='--')
                plt.axhline(1, color='r', linestyle='--')

        if file_path is not None:
            plt.savefig(file_path)
        plt.show()

    # Initializing log variables
    def _log_init(self):
        self.history = {}
        self.history['loss'] = []

        # Timestamp records
        self.pre_time = datetime.now() # Start timestamp
        self.start_time = self.pre_time
        self.history['time'] = [self.pre_time]

        for metric in self.metrics:
            if metric in ['acc', 'accuracy']:
                self.history['acc'] = []

    # log loss value and metrics
    def _log(self, t, y, p):
        # Calculating loss and metrics value
        self.history['loss'].append(np.mean(self.loss(y, p)))
        for metric in self.metrics:
            if metric in ['acc', 'accuracy']:
                acc = accuracy_score(np.argmax(y, 1), np.argmax(p, 1))
                self.history['acc'].append(acc)

        # Calculating timestamp
        current_time = datetime.now()
        time_diff = current_time - self.pre_time
        self.pre_time = current_time
        self.history['time'].append(self.pre_time)

        # Show loss and metrics value
        if self.verbose == 0:
            pass

        elif self.verbose == 1:
            message = 'Epoch {} - loss {:.3e}'.format(t, self.history['loss'][-1])

            for metric in self.metrics:
                if metric in ['acc', 'accuracy']:
                    message += ' - acc {:.2f}%'.format(self.history['acc'][-1] * 100)

            message += ' - {:.4f}s'.format(time_diff.total_seconds())
            percent = int(t / self.epochs * 100)
            sys.stdout.write('\r')
            sys.stdout.write('PROGRESS {0} {1}% [{2:<50s}] {3}'.format(to_timestamp(current_time - self.start_time), percent, '=' * int(percent / 2), message))
            sys.stdout.flush()

        elif self.verbose == 2:
            print('{} Epoch {} - loss {:.3e}'.format(to_timestamp(current_time - self.start_time), t, self.history['loss'][-1]), end='')

            for metric in self.metrics:
                if metric in ['acc', 'accuracy']:
                    print(' - acc {:.2f}%'.format(self.history['acc'][-1] * 100), end='')
            print(' - {:.4f}s'.format(time_diff.total_seconds()), end='')
            print()

        else:
            raise Exception('Verbose value must be 0, 1 or 2')

    # Input and output dimension check
    def _check_dim(self, x, y=None):
        if self.inputs != x.shape[1]:
            raise Exception('Input size is {}, but find {}'.format(self.inputs, x.shape[1]))

        if y is not None and self.outputs != y.shape[1]:
            raise Exception('Output size is {}, but find {}'.format(self.outputs, y.shape[1]))

    # Use python pickle to save model
    def save(self, file_path, scaler=None):
        if not self.hasCompile:
            raise Exception('Must "compile" before save')

        with open(file_path, 'wb') as file:
            pickle.dump((self, scaler), file, pickle.HIGHEST_PROTOCOL)

    # Use python pickle to load model
    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


class Scaler:
    def __init__(self, x_scaler=None, y_scaler=None, x_pca=None):
        self.x_scaler = None
        self.y_scaler = None
        self.x_decompose = None

        # Setting scaler
        if x_scaler is None:
            pass
        elif x_scaler == 'min_max':
            self.x_scaler = (x_scaler, MinMaxScaler())
        elif x_scaler == 'std':
            self.x_scaler = (x_scaler, StandardScaler())
        elif x_scaler == 'one_hot':
            self.x_scaler = (x_scaler, OneHotEncoder())
        else:
            raise Exception('Unknown x_scaler: "{}"'.format(x_scaler))

        if y_scaler is None:
            pass
        elif y_scaler == 'min_max':
            self.y_scaler = (y_scaler, MinMaxScaler())
        elif y_scaler == 'std':
            self.y_scaler = (y_scaler, StandardScaler())
        elif y_scaler == 'one_hot':
            self.y_scaler = (y_scaler, OneHotEncoder())
        else:
            raise Exception('Unknown y_scaler: "{}"'.format(y_scaler))

        if x_pca is not None:
            if x_pca == 'mle':
                self.x_decompose = PCA(n_components='mle', svd_solver='full')
            elif type(x_pca) is float:
                self.x_decompose = PCA(n_components=x_pca, svd_solver='full')
            else:
                self.x_decompose = PCA(n_components=x_pca)



    def x_fit(self, x):
        x_tmp = x
        if self.x_scaler is not None:
            name, scaler = self.x_scaler
            scaler.fit(x_tmp)

            if self.x_decompose is not None:
                if name == 'one_hot':
                    x_tmp = scaler.transform(x_tmp).toarray()
                else:
                    x_tmp = scaler.transform(x_tmp)

        if self.x_decompose is not None:
            self.x_decompose.fit(x_tmp)

    def y_fit(self, y):
        if self.y_scaler is not None:
            name, scaler = self.y_scaler
            scaler.fit(y)

    def fit(self, x, y):
        self.x_fit(x)
        self.y_fit(y)

    def x_transform(self, x):
        x_tmp = x
        if self.x_scaler is not None:
            name, scaler = self.x_scaler

            if name == 'one_hot':
                x_tmp = scaler.transform(x_tmp).toarray()
            else:
                x_tmp = scaler.transform(x_tmp)

        if self.x_decompose is not None:
            x_tmp = self.x_decompose.transform(x_tmp)

        return x_tmp

    def y_transform(self, y):
        y_tmp = y
        if self.y_scaler is not None:
            name, scaler = self.y_scaler
            if name == 'one_hot':
                y_tmp = scaler.transform(y_tmp).toarray()
            else:
                y_tmp = scaler.transform(y_tmp)

        return y_tmp

    def transform(self, x, y):
        return self.x_transform(x), self.y_transform(y)

    def y_inverse(self, y):
        y_tmp = y
        if self.y_scaler is not None:
            name, scaler = self.y_scaler

            if name == 'one_hot':
                y_tmp = y_tmp.dot(scaler.active_features_)
            elif scaler:
                y_tmp = scaler.inverse_transform(y_tmp)

        return y_tmp

    @property
    def decompose_ratio(self):
        if self.x_decompose is None:
            return ''
        else:
            message = '['
            for ratio in self.x_decompose.explained_variance_ratio_:
                message += '{:>4.0f}%'.format(ratio*100)
            return message + ']'