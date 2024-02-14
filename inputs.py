import numpy as np
import matplotlib.pyplot as plt


def build_training(n_samples):

    # xtrain = [num_inputs * num_samples]
    # ytrain = [num_outputs * num_samples]
    x_train = np.random.rand(2, n_samples)
    y_train = np.zeros((2, n_samples), dtype=np.float32)


    for i in range(n_samples):

        x_coord = x_train[0,i]
        y_coord = 0.02660099 + 4.171921*x_coord - 9.371921*x_coord**2 + 5*x_coord**3

        if (y_coord > x_train[1,i]):
            y_train[0,i] = 0   # safe
            y_train[1,i] = 1
        else:
            y_train[0,i] = 1   # dangerous
            y_train[1,i] = 0

    return x_train, y_train


def save_training(x_train, y_train):

    np.savetxt('x_train.dat', x_train)
    np.savetxt('y_train.dat', y_train)

    return

def load_training():

    x_train = np.loadtxt('x_train.dat', dtype=np.float32)
    y_train = np.loadtxt('y_train.dat', dtype=np.float32)

    return x_train, y_train


def plot_map(n_samples, x_train, y_train):

    for i in range(n_samples):
        if ((y_train[0,i] == 0) and (y_train[1,i] == 1)):   # safe
            plt.scatter(x_train[0,i], x_train[1,i], color='green', edgecolor='black')
        else:                  # dangerous
            plt.scatter(x_train[0,i], x_train[1,i], color='red', edgecolor='black')

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('x [-]')
    plt.ylabel('y [-]')
    plt.show()






