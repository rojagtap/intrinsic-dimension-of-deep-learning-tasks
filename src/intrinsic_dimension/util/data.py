from torchvision import datasets
from torchvision.transforms import ToTensor


def get_dataset(name):
    """
    utility to download and return standard datasets
    """

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

        return train_dataset, test_dataset

    return None, None
