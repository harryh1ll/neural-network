import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:
    def __init__(self, input_size, output_size):
        self.w = np.random.rand(output_size, input_size)
        self.b = np.zeros(output_size).reshape(output_size, 1)
        self.inputs = None

    def forward(self, inputs):
        self.z = np.matmul(self.w, inputs) + self.b
        a = sigmoid(self.z)

        # cache inputs for use in backpropagation
        self.inputs = inputs

        return a

    def backward(self, pass_back, learning_rate):
        error = pass_back * sigmoid_derivative(self.z)
        grad_w = error * self.inputs.T
        grad_b = error
        pass_back = np.matmul(self.w.T, error)

        self.w = self.w - (learning_rate * grad_w)
        self.b = self.b - (learning_rate * grad_b)

        return pass_back




