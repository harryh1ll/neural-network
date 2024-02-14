import sys
import numpy as np
import matplotlib.pyplot as plt
from layer import Layer


class NeuralNetwork:

    def __init__(self, neurons_per_layer):
        self.n_layers = len(neurons_per_layer)-1
        self.layers = []

        for i in range(self.n_layers):
            self.layers.append(Layer(neurons_per_layer[i], neurons_per_layer[i+1]))


    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, inputs, cost_gradient, learning_rate, m):
        pass_back = cost_gradient
        for layer in reversed(self.layers):
            pass_back = layer.backward(pass_back, learning_rate)


    def train(self, inputs, targets, epochs, learning_rate, num_samples):
        iteration = []
        mse_total = []

        for epoch in range(epochs):
            mse = 0

            for sample in range(num_samples):
                single_input = inputs[:, sample].reshape(2, 1)
                predictions = self.forward(single_input)

                single_target = targets[:, sample].reshape(2, 1)
                cost_gradient = (predictions - single_target)

                mse += np.mean((predictions - single_target)**2)

                self.backward(single_input, cost_gradient, learning_rate, num_samples)

            if epoch % 100 == 0:
                print(f'{epoch}  {mse/num_samples}')
                iteration.append(epoch)
                mse_total.append(mse/num_samples)

        plt.plot(iteration, mse_total)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.show()


    def test(self, inputs, targets, num_samples):
        count = 0
        for sample in range(num_samples):
            single_input = inputs[:, sample].reshape(2, 1)
            predictions = self.forward(single_input)

            rounded_pred = np.round(predictions)
            single_tar = targets[:, sample].reshape(2, 1)
            if np.array_equal(rounded_pred, single_tar):
                count += 1

        print(num_samples, count)


