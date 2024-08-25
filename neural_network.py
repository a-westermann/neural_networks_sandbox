import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate, shape):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.weights = np.random.random(shape)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def load_weights_bias(self, weight_file, bias_file):
        # self.weights = np.fromstring(open(weight_file).read(), float)
        # self.bias = np.fromstring(open(bias_file).read(), float)
        # np.reshape(self.weights, (28, 28))
        # np.reshape(self.bias, (28, 28))
        self.weights = eval(open(weight_file).read())
        self.bias = eval(open(bias_file).read())

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self.sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        return derror_dbias, derror_dweights

    def update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)
