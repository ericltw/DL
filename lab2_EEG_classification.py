import argparse
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='EEG')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--learning_rate', default=1e-2)
    parser.add_argument('--epochs', default=300)

    return parser.parse_args()


def read_bci_data():
    S4b_train = np.load('./lab2_data/S4b_train.npz')
    X11b_train = np.load('./lab2_data/X11b_train.npz')
    S4b_test = np.load('./lab2_data/S4b_test.npz')
    X11b_test = np.load('./lab2_data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(f'Input shape: {train_data.shape}, {train_label.shape}, {test_data.shape}, {test_label.shape}')

    return train_data, train_label, test_data, test_label


class EEGNet(nn.Module):
    def __init__(self, activation_function: nn.modules.activation):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False,
            ),
            nn.BatchNorm2d(
                16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        )

        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            activation_function(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0
            ),
            nn.Dropout(p=0.25)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False,
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            activation_function(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True,
            )
        )

    def forward(self, inputs: TensorDataset):
        inputs = self.first_conv(inputs)
        inputs = self.depth_wise_conv(inputs)
        inputs = self.separable_conv(inputs)
        return self.classify(inputs)


class DeepConvNet(nn.Module):
    def __init__(self, activation_function: nn.modules.activation):
        super(DeepConvNet, self).__init__()

        self.deepconv = [25, 50, 100, 200]
        dropout = 0.5

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.deepconv[0],
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 0),
                bias=True
            ),
            nn.Conv2d(
                in_channels=self.deepconv[0],
                out_channels=self.deepconv[0],
                kernel_size=(2, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True
            ),
            nn.BatchNorm2d(
                self.deepconv[0]
            ),
            activation_function(),
            nn.MaxPool2d(
                kernel_size=(1, 2)
            ),
            nn.Dropout(
                p=dropout
            )
        )

        for index in range(1, len(self.deepconv)):
            setattr(self, 'conv' + str(index), nn.Sequential(
                nn.Conv2d(
                    in_channels=self.deepconv[index - 1],
                    out_channels=self.deepconv[index],
                    kernel_size=(1, 5),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=True
                ),
                nn.BatchNorm2d(
                    self.deepconv[index]
                ),
                activation_function(),
                nn.MaxPool2d(
                    kernel_size=(1, 2)
                ),
                nn.Dropout(
                    p=dropout
                )
            ))

        flatten_size = self.deepconv[-1] * reduce(
            lambda x, _: round((x - 4) / 2), self.deepconv, 750)
        self.classify = nn.Sequential(
            nn.Linear(flatten_size, 2, bias=True),
        )

    def forward(self, inputs: TensorDataset):
        for index in range(len(self.deepconv)):
            inputs = getattr(self, 'conv' + str(index))(inputs)

        # Flatten
        inputs = inputs.view(-1, self.classify[0].in_features)
        inputs = self.classify(inputs)
        return inputs


def show_result(model, epochs, accuracy, model_names):
    # Plot
    plt.figure(0)
    if model == 'EEG':
        plt.title('EEGNet / Activation Function Comparison')
    else:
        plt.title('DeepConvNet / Activation Function Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    for type_of_accuracy, accuracy_dict in accuracy.items():
        for model in model_names:
            plt.plot(range(epochs), accuracy_dict[model], label=f'{model}_{type_of_accuracy}')
            print(f'{model}_{type_of_accuracy}: {max(accuracy_dict[model]):.2f} %')

    plt.legend(loc='lower right')
    plt.show()


def train(model, batch_size, learning_rate, epochs, device, train_dataset, test_dataset):
    if model == 'EEG':
        models = {
            'EEG_ReLU': EEGNet(nn.ReLU).to(device),
            'EEG_LeakyReLU': EEGNet(nn.LeakyReLU).to(device),
            'EEG_ELU': EEGNet(nn.ELU).to(device),
        }
    else:
        models = {
            'DeepConvNet_ReLU': DeepConvNet(nn.ReLU).to(device),
            'DeepConvNet_LeakyReLU': DeepConvNet(nn.LeakyReLU).to(device),
            'DeepConvNet_ELU': DeepConvNet(nn.ELU).to(device),
        }

    # Init accuracy data structure for recording accuracy data.
    model_names = [f'{model}_ReLU', f'{model}_LeakyReLU', f'{model}_ELU']
    accuracy = {
        'train': {model_name: [0 for _ in range(epochs)] for model_name in model_names},
        'test': {model_name: [0 for _ in range(epochs)] for model_name in model_names}
    }

    # Init data loader.
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    # Init loss function.
    loss_function = nn.CrossEntropyLoss()

    # Start training.
    # Train and test model one by one.
    for model_name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(f'Training: {model_name}')

        for epoch in tqdm(range(epochs)):
            # Sets the module in training mode.
            model.train()

            for data_batch, label_batch in train_loader:
                # Put data and labels. to GPU.
                data = data_batch.to(device)
                labels = label_batch.to(device).long()

                # Get predict labels.
                predict_labels = model.forward(inputs=data)
                # Compute loss based on cross entropy.
                loss = loss_function(predict_labels, labels)

                # Execute backward propagation.
                loss.backward()

                # Renew optimizer.
                optimizer.step()
                optimizer.zero_grad()

                # Compute the total number of right predictions for this batch.
                accuracy['train'][model_name][epoch] += (torch.max(predict_labels, 1)[1] == labels).sum().item()
            # Compute the accuracy for this epoch.
            accuracy['train'][model_name][epoch] = 100.0 * accuracy['train'][model_name][epoch] / len(train_dataset)

            # Sets the module in evaluation mode
            model.eval()
            with torch.no_grad():
                for data_batch, label_batch in test_loader:
                    # Put data and labels. to GPU.
                    data = data_batch.to(device)
                    labels = label_batch.to(device).long()

                    # Get predict labels.
                    predict_labels = model.forward(inputs=data)

                    # Compute the total number of right predictions for this batch.
                    accuracy['test'][model_name][epoch] += (torch.max(predict_labels, 1)[1] == labels).sum().item()
                # Compute the accuracy for this epoch.
                accuracy['test'][model_name][epoch] = 100.0 * accuracy['test'][model_name][epoch] / len(train_dataset)

        print()
        torch.cuda.empty_cache()

    show_result(model, epochs, accuracy, model_names)


def main():
    # Parse input arguments.
    args = parse_argument()
    model = args.model
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)

    # Prepare training, testing data.
    train_data, train_label, test_data, test_label = read_bci_data()
    # Prepare dataset in TensorDataset type.
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    # Get device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train(model, batch_size, learning_rate, epochs, device, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
