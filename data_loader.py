import numpy as np
from torchvision import datasets
import torch.nn.functional as F


def load_mnist_data_torch(label_data_rate, seed):
    """
    load MNIST dataset。

    Args:
        - label_data_rate: 0-1

    Returns:
        - x_label, y_label: labeled data
        - x_unlab: unlabeled data
        - x_test, y_test: test data
    """

    # load MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    x_train = mnist_train.data.view(-1, 28 * 28).float() / 255.0  # [n, 784]
    y_train = mnist_train.targets  # [n]
    x_test = mnist_test.data.view(-1, 28 * 28).float() / 255.0  # [n, 784]
    y_test = mnist_test.targets  # [n]

    y_train_oh = F.one_hot(y_train, num_classes=10).float()
    y_test_oh = F.one_hot(y_test, num_classes=10).float()

    np.random.seed(seed)
    idx = np.random.permutation(len(y_train))
    label_idx = idx[:int(label_data_rate * len(idx))]
    unlab_idx = idx[int(label_data_rate * len(idx)):]

    # Labeled data
    x_label = x_train[label_idx]
    y_label = y_train_oh[label_idx]

    # Unlabeled data
    x_unlab = x_train[unlab_idx]

    return x_label.numpy(), y_label.numpy(), x_unlab.numpy(), x_test.numpy(), y_test_oh.numpy()





