import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type')
    parser.add_argument('--activation_function', default='sigmoid')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--num_of_hidden_neurals', default=4)

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
    def init_weight(self):
        return np.random.uniform(0, 1, (self.num_of_neurals, self.num_of_next_layer_neurals))

    def __init__(self, num_of_neurals, num_of_next_layer_neurals, activation_function, learning_rate):
        self.num_of_neurals = num_of_neurals
        self.num_of_next_layer_neurals = num_of_next_layer_neurals
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.weight = self.init_weight()
        self.forward_output = []
        self.forward_gradient = []
        self.backward_gradient = []

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        return np.multiply(x, 1.0 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def derivative_relu(x):
        return np.heaviside(x, 0.0)

    def forward(self, inputs):
        self.forward_gradient = inputs

        if self.activation_function == 'sigmoid':
            self.forward_output = self.sigmoid(np.dot(inputs, self.weight))
        elif self.activation_function == 'relu':
            self.forward_output = self.relu(np.dot(inputs, self.weight))
        elif self.activation_function == 'none':
            self.forward_output = np.dot(inputs, self.weight)

        return self.forward_output

    def backward(self, derivative_loss):
        # Compute ∂C/∂Z'
        if self.activation_function == 'sigmoid':
            self.backward_gradient = derivative_loss * self.derivative_sigmoid(self.forward_output)
        elif self.activation_function == 'relu':
            self.backward_gradient = derivative_loss * self.derivative_relu(self.forward_output)
        elif self.activation_function == 'none':
            self.backward_gradient = derivative_loss

        # return w5*(∂C/∂Za) + w6*(∂X/∂Zb) + ...
        return np.dot(self.backward_gradient, self.weight.T)

    def update(self):
        gradient = np.dot(self.forward_gradient.T, self.backward_gradient)

        # Update weight.
        self.weight -= self.learning_rate * gradient


class NNetwork:
    def init_layers(self):
        # Init input layers.
        layers = [Layer(self.num_of_input_neurals, self.num_of_hidden_neurals, self.activation_function,
                        self.learning_rate)]

        # Init hidden layers.
        for _ in range(self.num_of_hidden_layers - 1):
            layers.append(
                Layer(self.num_of_hidden_neurals, self.num_of_hidden_neurals, self.activation_function,
                      self.learning_rate))

        # Init output layers.
        layers.append(Layer(self.num_of_hidden_neurals, self.num_of_output_neurals, 'sigmoid',
                            self.learning_rate))

        return layers

    def __init__(self, inputs, labels, activation_function, learning_rate, num_of_hidden_layers=2,
                 num_of_input_neurals=2, num_of_hidden_neurals=4, num_of_output_neurals=1, epoch=100000):
        self.inputs = inputs
        self.labels = labels
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.num_of_hidden_layers = num_of_hidden_layers
        # The number of neurals of input, hidden, and output layer.
        self.num_of_input_neurals = num_of_input_neurals
        self.num_of_hidden_neurals = num_of_hidden_neurals
        self.num_of_output_neurals = num_of_output_neurals
        self.epoch = epoch

        self.layers = self.init_layers()
        self.predict = []
        self.losses = []

    @staticmethod
    def compute_error(x1, x2):
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

    def verify_prediction(self):
        for i, predict_val in enumerate(self.predict):
            predict_result = 1 if predict_val[0] >= 0.5 else 0

            if predict_result != self.labels[i]:
                return False

        return True

    def train(self):
        losses = []

        for i in range(self.epoch):
            self.predict = self.forward()
            loss = self.MSE(self.compute_error(self.predict, self.labels))
            derivative_MSE = self.derivative_MSE(self.compute_error(self.predict, self.labels))
            self.backward(derivative_MSE)
            self.update_weight()

            print(f'epoch: {i + 1} loss: {loss}')
            self.losses.append(loss)
            if self.verify_prediction():
                break

    def print_prediction(self):
        print('##### Prediction')
        for predict_val in self.predict:
            print(f'{float(predict_val):.10f}')

    def get_accuracy(self):
        correct = 0
        for i, predict_val in enumerate(self.predict):
            predict_result = 1 if predict_val[0] >= 0.5 else 0
            if predict_result == self.labels[i]:
                correct += 1

        return correct / len(self.labels)

    def print_statistics(self):
        print('##### Statistics')
        print(f'Number of hidden layers: {self.num_of_hidden_layers}')
        print(f'Number of input neurals: {self.num_of_input_neurals}')
        print(f'Number of hidden neurals: {self.num_of_hidden_neurals}')
        print(f'Number of output neurals: {self.num_of_output_neurals}')
        print(f'Activation function: {self.activation_function}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Accuracy: {self.get_accuracy()}')

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
        self.print_prediction()
        self.print_statistics()
        self.draw_result(self.inputs, self.labels, self.predict, self.losses)


def main():
    args = parse_arguments()
    data_type = int(args.data_type)
    activation_function = args.activation_function
    learning_rate = float(args.learning_rate)
    num_of_hidden_neurals = int(args.num_of_hidden_neurals)

    inputs, labels = generate_data(data_type)

    network = NNetwork(inputs, labels, activation_function, learning_rate, num_of_hidden_neurals=num_of_hidden_neurals)
    network.train()
    network.show_result()


if __name__ == '__main__':
    main()
