import argparse
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
# from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as torch_models
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=5e-4)

    return parser.parse_args()


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('./lab3_data/train_img.csv')
        label = pd.read_csv('./lab3_data/train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('./lab3_data/test_img.csv')
        label = pd.read_csv('./lab3_data/test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.image_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.image_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.image_name)

    def __getitem__(self, index):
        # Load image.
        image_path = os.path.join(self.root, f'{self.image_name[index]}.jpeg')
        # image = mpimg.imread(image_path)
        image = PIL.Image.open(image_path)

        # Get the ground truth label.
        label = self.label[index]

        # Transform the .jpeg rgb images during the training phase.
        # Convert the pixel value to [0, 1].
        # image = np.where(image < 128, 0, 1)
        # Transpose the image shape from [H, W, C] to [C, H, W].
        # image = np.transpose(image, (2, 0, 1))
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])
        image = transform(image)

        return image, label


# Basic block for building ResNet18.
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU(
            inplace=True,
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
        )
        self.down_sample = down_sample

    def forward(self, inputs):
        outputs = self.block(inputs)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        outputs = self.activation(outputs + inputs)

        return outputs


# Basic block for building ResNet50.
class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(BottleneckBlock, self).__init__()

        external_channels = out_channels * self.expansion
        self.activation = nn.ReLU(
            inplace=True,
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=external_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=external_channels,
            ),
        )
        self.down_sample = down_sample

    def forward(self, inputs):
        outputs = self.block(inputs)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        outputs = self.activation(outputs + inputs)

        return outputs


class ResNet(nn.Module):
    @staticmethod
    def get_pretrained_weights(architecture):
        if architecture == 'resnet18':
            return ResNet18_Weights.IMAGENET1K_V1
        else:
            return ResNet50_Weights.IMAGENET1K_V1

    # Generate blocks for conv_2, conv_3, conv_4, and conv_5.
    def generate_blocks(self, block, num_of_blocks, in_channels, stride=1):
        down_sample = None

        # If the number of channels, width, or height will be changed, do down_sample for residual.
        if stride != 1 or self.current_channels != in_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.current_channels,
                    out_channels=in_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=in_channels * block.expansion,
                ),
            )

        # Build first block for conv_2, conv_3, conv_4, and conv_5.
        layers = [
            block(
                in_channels=self.current_channels,
                out_channels=in_channels,
                stride=stride,
                down_sample=down_sample,
            ),
        ]

        # After building the first block, the current_channels will be changed (because of convolution
        # in first block).
        self.current_channels = in_channels * block.expansion
        # Build remaining blocks for conv_2, conv_3, conv_4, and conv_5.
        layers += [
            block(
                in_channels=self.current_channels,
                out_channels=in_channels,
            ) for _ in range(1, num_of_blocks)]

        return nn.Sequential(*layers)

    def __init__(self, architecture, block, layers, pretrain: bool):
        super(ResNet, self).__init__()

        if pretrain:
            pretrained_weights = self.get_pretrained_weights(architecture)
            pretrained_resnet = getattr(torch_models, architecture)(weights=pretrained_weights)
            self.conv_1 = nn.Sequential(
                getattr(pretrained_resnet, 'conv1'),
                getattr(pretrained_resnet, 'bn1'),
                getattr(pretrained_resnet, 'relu'),
                getattr(pretrained_resnet, 'maxpool')
            )

            # Layers
            self.conv_2 = getattr(pretrained_resnet, 'layer1')
            self.conv_3 = getattr(pretrained_resnet, 'layer2')
            self.conv_4 = getattr(pretrained_resnet, 'layer3')
            self.conv_5 = getattr(pretrained_resnet, 'layer4')

            self.classify = nn.Sequential(
                getattr(pretrained_resnet, 'avgpool'),
                nn.Flatten(),
                nn.Linear(getattr(pretrained_resnet, 'fc').in_features, out_features=50),
                nn.ReLU(
                    inplace=True,
                ),
                nn.Dropout(
                    p=0.25,
                ),
                nn.Linear(
                    in_features=50,
                    out_features=5,
                ),
            )

            del pretrained_resnet
        else:
            self.current_channels = 64

            self.conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=64,
                ),
                nn.ReLU(
                    inplace=True,
                ),
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )

            # Layers
            self.conv_2 = self.generate_blocks(
                block=block,
                num_of_blocks=layers[0],
                in_channels=64,
            )
            self.conv_3 = self.generate_blocks(
                block=block,
                num_of_blocks=layers[1],
                in_channels=128,
                stride=2,
            )
            self.conv_4 = self.generate_blocks(
                block=block,
                num_of_blocks=layers[2],
                in_channels=256,
                stride=2,
            )
            self.conv_5 = self.generate_blocks(
                block=block,
                num_of_blocks=layers[3],
                in_channels=512,
                stride=2,
            )
            self.classify = nn.Sequential(
                nn.AdaptiveAvgPool2d(
                    output_size=(1, 1),
                ),
                nn.Flatten(),
                nn.Linear(
                    in_features=512 * block.expansion,
                    out_features=50,
                ),
                nn.ReLU(
                    inplace=True,
                ),
                nn.Dropout(
                    p=0.25,
                ),
                nn.Linear(
                    in_features=50,
                    out_features=5,
                ),
            )

    def forward(self, inputs: TensorDataset):
        partial_results = inputs
        for idx in range(1, 6):
            partial_results = getattr(self, f'conv_{idx}')(partial_results)
        return self.classify(partial_results)


def resnet_18(pretrain=False):
    return ResNet(
        architecture='resnet18',
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        pretrain=pretrain,
    )


def resnet_50(pretrain=False):
    return ResNet(
        architecture='resnet50',
        block=BottleneckBlock,
        layers=[3, 4, 6, 3],
        pretrain=pretrain,
    )


def show_results(target_model, epochs, accuracy, prediction, ground_truth, model_names):
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # Plot
    plt.figure(0)
    plt.title(f'Result Comparison ({target_model})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    # Plot comparison graph.
    for type_of_accuracy, accuracy_dict in accuracy.items():
        for model in model_names:
            plt.plot(range(epochs), accuracy_dict[model], label=f'{model}_{type_of_accuracy}')
            print(f'{model}_{type_of_accuracy}: {max(accuracy_dict[model]):.2f} %')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./results/{target_model}_comparison.png')
    plt.close()

    # Plot confusion matrix.
    for key, predict_labels in prediction.items():
        cm = confusion_matrix(
            y_true=ground_truth,
            y_pred=predict_labels,
            normalize='true'
        )
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
        plt.title(f'Normalized confusion matrix ({key})')
        plt.tight_layout()
        plt.savefig(f'./results/{key.replace(" ", "_").replace("/", "_")}_confusion.png')
        plt.close()


def train(target_model, batch_size, learning_rate, epochs, momentum, weight_decay, device, train_dataset, test_dataset):
    # Init model_names, models and data structure of accuracy.
    if target_model == 'ResNet18':
        model_names = [
            'ResNet18(wo_pretraining)',
            'ResNet18(w_pretraining)'
        ]
        models = {
            model_names[0]: resnet_18().to(device),
            model_names[1]: resnet_18(pretrain=True).to(device)
        }
    else:
        model_names = [
            'ResNet50(wo_pretraining)',
            'ResNet50(w_pretraining)'
        ]
        models = {
            model_names[0]: resnet_50().to(device),
            model_names[1]: resnet_50(pretrain=True).to(device)
        }
    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in model_names},
        'test': {key: [0 for _ in range(epochs)] for key in model_names}
    }

    # Init prediction data structure.
    prediction = {key: None for key in model_names}

    # Init data loader.
    print('Init dataloader...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Init ground truth data structure.
    ground_truth = np.array([], dtype=int)
    for _, label in test_loader:
        ground_truth = np.concatenate((ground_truth, label.long().view(-1).numpy()))

    # Init loss function.
    loss_function = nn.CrossEntropyLoss()

    # Start training.
    # Train and test model one by one.
    for model_name, model in models.items():
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        max_test_accuracy = 0

        print(f'Training: {model_name}')
        for epoch in tqdm(range(epochs)):
            # Sets the module in training mode.
            model.train()

            for data_batch, label_batch in tqdm(train_loader):
                # Put data and labels to GPU.
                data = data_batch.to(device)
                labels = label_batch.to(device).long().view(-1)

                # Get predict labels.
                predict_labels = model.forward(inputs=data)
                # Compute loss based on cross entropy.
                loss = loss_function(predict_labels, labels)

                # Execute backward propagation.
                loss.backward()

                # Renew optimizer.
                optimizer.step()
                optimizer.zero_grad()

                accuracy['train'][model_name][epoch] += (torch.max(predict_labels, 1)[1] == labels).sum().item()
            accuracy['train'][model_name][epoch] = 100.0 * accuracy['train'][model_name][epoch] / len(train_dataset)

            # Sets the module in evaluation mode
            model.eval()
            with torch.no_grad():
                predict_labels = np.array([], dtype=int)

                for data_batch, label_batch in tqdm(test_loader):
                    # Put data and labels. to GPU.
                    data = data_batch.to(device)
                    labels = label_batch.to(device).long().view(-1)

                    # Get predict labels.
                    outputs = model.forward(inputs=data)
                    outputs = torch.max(outputs, 1)[1]
                    predict_labels = np.concatenate((predict_labels, outputs.cpu().numpy()))

                    # Compute the total number of right predictions for this batch.
                    accuracy['test'][model_name][epoch] += (outputs == labels).sum().item()

                # Compute the accuracy for this epoch.
                accuracy['test'][model_name][epoch] = 100.0 * accuracy['test'][model_name][epoch] / len(test_dataset)

                # Save max_accuracy and the result of prediction, prediction will be used for drawing confusion matrix.
                if accuracy['test'][model_name][epoch] > max_test_accuracy:
                    max_test_accuracy = accuracy['test'][model_name][epoch]
                    prediction[model_name] = predict_labels

                # Save model.
                if accuracy['test'][model_name][epoch] > 80:
                    torch.save(model.state_dict(), f'C:/Users/user/Documents/Course/DL/{model_name}_batch_size_'
                                                   f'{batch_size}_lr_{learning_rate}_epoch_{epochs}_momentum_'
                                                   f'{momentum}_weight_decay_{weight_decay}.pt')
    # Show results.
    show_results(target_model, epochs, accuracy, prediction, ground_truth, model_names)


def main():
    # Parse input arguments.
    args = parse_argument()
    model = args.model
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    momentum = float(args.momentum)
    weight_decay = float(args.weight_decay)

    # Prepare dataset.
    print('Prepare dataset')
    train_dataset = RetinopathyDataset('./lab3_data/data', 'train')
    test_dataset = RetinopathyDataset('./lab3_data/data', 'test')

    # Get device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train(model, batch_size, learning_rate, epochs, momentum, weight_decay, device, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
