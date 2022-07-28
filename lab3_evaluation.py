import argparse
from lab3_retinopathy_detection import RetinopathyDataset, resnet_18
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name')

    return parser.parse_args()


def evaluate(file_name, device, train_dataset, test_dataset):
    model = resnet_18(pretrain=True).to(device)
    model.load_state_dict(torch.load(file_name))

    # Init accuracy data structure for recording accuracy data.
    accuracy = {
        'train': 0,
        'test': 0,
    }

    # Init data loader.
    train_loader = DataLoader(train_dataset, 32)
    test_loader = DataLoader(test_dataset, 32)

    # Sets the module in evaluation mode
    model.eval()
    with torch.no_grad():
        for data_batch, label_batch in tqdm(train_loader):
            # Put data and labels. to GPU.
            data = data_batch.to(device)
            labels = label_batch.to(device).long().view(-1)

            # Get predict labels.
            outputs = model.forward(inputs=data)
            outputs = torch.max(outputs, 1)[1]

            # Compute the total number of right predictions for this batch.
            accuracy['train'] += (outputs == labels).sum().item()

        # Compute the accuracy for this epoch.
        accuracy['train'] = 100.0 * accuracy['train'] / len(train_dataset)

    # Sets the module in evaluation mode
    model.eval()
    with torch.no_grad():
        for data_batch, label_batch in tqdm(test_loader):
            # Put data and labels. to GPU.
            data = data_batch.to(device)
            labels = label_batch.to(device).long().view(-1)

            # Get predict labels.
            outputs = model.forward(inputs=data)
            outputs = torch.max(outputs, 1)[1]

            # Compute the total number of right predictions for this batch.
            accuracy['test'] += (outputs == labels).sum().item()

        # Compute the accuracy for this epoch.
        accuracy['test'] = 100.0 * accuracy['test'] / len(test_dataset)

    print(f'Train accuracy: {accuracy["train"]}%')
    print(f'Test accuracy: {accuracy["test"]}%')


def main():
    args = parse_argument()
    file_name = args.file_name

    # Prepare dataset.
    print('Prepare dataset')
    train_dataset = RetinopathyDataset('./lab3_data/data', 'train')
    test_dataset = RetinopathyDataset('./lab3_data/data', 'test')

    # Get device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    evaluate(file_name, device, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
