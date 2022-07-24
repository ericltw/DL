import argparse
from lab2_EEG_classification import read_bci_data, EEGNet
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name')

    return parser.parse_args()


def evaluate(file_name, device, train_dataset, test_dataset):
    model = EEGNet(nn.LeakyReLU).to(device)
    model.load_state_dict(torch.load(file_name))

    # Init accuracy data structure for recording accuracy data.
    accuracy = {
        'train': 0,
        'test': 0,
    }

    # Init data loader.
    train_loader = DataLoader(train_dataset, len(train_dataset))
    test_loader = DataLoader(test_dataset, len(test_dataset))

    # Sets the module in evaluation mode
    model.eval()
    with torch.no_grad():
        for data_batch, label_batch in train_loader:
            # Put data and labels. to GPU.
            data = data_batch.to(device)
            labels = label_batch.to(device).long()

            # Get predict labels.
            predict_labels = model.forward(inputs=data)

            # Compute the total number of right predictions for this batch.
            accuracy['train'] += (torch.max(predict_labels, 1)[1] == labels).sum().item()

        # Compute the accuracy for this epoch.
        accuracy['train'] = 100.0 * accuracy['train'] / len(test_dataset)

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
            accuracy['test'] += (torch.max(predict_labels, 1)[1] == labels).sum().item()

        # Compute the accuracy for this epoch.
        accuracy['test'] = 100.0 * accuracy['test'] / len(test_dataset)

    print('Model: EEG_LeakyReLU')
    print(f'Train accuracy: {accuracy["train"]}%')
    print(f'Test accuracy: {accuracy["test"]}%')


def main():
    args = parse_argument()
    file_name = args.file_name

    # Prepare training, testing data.
    train_data, train_label, test_data, test_label = read_bci_data()
    # Prepare dataset in TensorDataset type.
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    # Get device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    evaluate(file_name, device, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
