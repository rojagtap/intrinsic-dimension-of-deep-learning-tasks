from torchvision import datasets
from torchvision.transforms import ToTensor


def get_dataset(name):
    """
    utility to download and return standard datasets
    """

    train_dataset, test_dataset = None, None
    if name == "mnist":
        train_dataset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

    return train_dataset, test_dataset
