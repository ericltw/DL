import argparse
import matplotlib.pyplot as plt
import numpy as np


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type')

    return parser.parse_args()


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def generate_data(data_type):
    if data_type == 0:
        return generate_linear()
    else:
        return generate_XOR_easy()


class Layer:
    def initWeight(self):
        return np.random.uniform(0, 1, (self.num_of_neurals, self.num_of_next_layer_neurals))

    def __init__(self, num_of_neurals, num_of_next_layer_neurals, learning_rate=0.025):
        self.num_of_neurals = num_of_neurals
        self.num_of_next_layer_neurals = num_of_next_layer_neurals
        self.learning_rate = learning_rate
        self.weight = self.initWeight()
        self.forward_output = []
        self.forward_gradient = []
        self.backward_gradient = []

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        result = np.multiply(x, 1.0 - x)
        return result

    def forward(self, inputs):
        self.forward_gradient = inputs
        self.forward_output = self.sigmoid(np.dot(inputs, self.weight))

        return self.forward_output

    def backward(self, derivative_loss):
        # Compute ∂C/∂Z'
        self.backward_gradient = derivative_loss * self.derivative_sigmoid(self.forward_output)

        # return w5*(∂C/∂Za) + w6*(∂X/∂Zb) + ...
        return np.dot(self.backward_gradient, self.weight.T)

    def update(self):
        gradient = np.dot(self.forward_gradient.T, self.backward_gradient)

        # Update weight.
        self.weight -= self.learning_rate * gradient


class NNetwork:
    def initLayers(self):
        # Init input layers.
        layers = [Layer(self.num_of_input_neurals, self.num_of_hidden_neurals)]

        # Init hidden layers.
        for _ in range(self.num_of_hidden_layers - 1):
            layers.append(Layer(self.num_of_hidden_neurals, self.num_of_hidden_neurals))

        # Init output layers.
        layers.append(Layer(self.num_of_hidden_neurals, self.num_of_output_neurals))

        return layers

    def __init__(self, inputs, labels, num_of_hidden_layers=2, num_of_input_neurals=2, num_of_hidden_neurals=4,
                 num_of_output_neurals=1, epoch=1000000):
        self.inputs = inputs
        self.labels = labels
        self.num_of_hidden_layers = num_of_hidden_layers
        # The number of neurals of input, hidden, and output layer.
        self.num_of_input_neurals = num_of_input_neurals
        self.num_of_hidden_neurals = num_of_hidden_neurals
        self.num_of_output_neurals = num_of_output_neurals
        self.epoch = epoch

        self.layers = self.initLayers()
        self.predict = []
        self.losses = []

    @staticmethod
    def computeError(x1, x2):
        return x1 - x2

    def forward(self):
        inputs = self.inputs

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    @staticmethod
    def MSE(error):
        loss = 0
        error = error.flatten()

        for j in range(error.size):
            loss += error[j] ** 2
        return loss / error.size

    @staticmethod
    def derivative_MSE(error):
        return 2 * error

    def backward(self, derivative_loss):
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update_weight(self):
        for layer in self.layers:
            layer.update()

    @staticmethod
    def draw_result(x, y, pred_y, losses):
        plt.subplot(1, 3, 1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.subplot(1, 3, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] < 0.5:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.subplot(1, 3, 3)
        plt.title('Learning Curve', fontsize=18)
        plt.plot(losses)

        plt.show()

    def show_result(self):
        self.draw_result(self.inputs, self.labels, self.predict, self.losses)
        plt.show()

    def train(self):
        losses = []

        for i in range(self.epoch):
            self.predict = self.forward()
            loss = self.MSE(self.computeError(self.predict, self.labels))
            derivative_MSE = self.derivative_MSE(self.computeError(self.predict, self.labels))
            self.backward(derivative_MSE)
            self.update_weight()

            print(f'epoch: {i + 1} loss: {loss}')
            self.losses.append(loss)
            if loss < 1e-4:
                break


def main():
    args = parseArguments()
    data_type = int(args.data_type)

    inputs, labels = generate_data(data_type)

    network = NNetwork(inputs, labels)
    network.train()
    network.show_result()


if __name__ == '__main__':
    main()
