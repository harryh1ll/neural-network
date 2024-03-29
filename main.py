import sys
from src.inputs import *
from src.neural_network import NeuralNetwork


# Example usage
num_samples = 100
learning_rate = 0.1
epochs = 100000

x_train, y_train = build_training(num_samples)
# x_train, y_train = load_training()
# plot_map(num_samples, x_train, y_train)

model = NeuralNetwork([2, 5, 2])
model.train(x_train, y_train, epochs, learning_rate, num_samples)

num_test = 50
x_test, y_test = build_training(num_test)
model.test(x_test, y_test, num_test)