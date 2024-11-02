import torch


class LeNet5(torch.nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0):
        super(LeNet5, self).__init__()

        self.input_dim = input_size[0]
        self.num_classes = num_classes

        # Convolutional stack (features)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size[-1], out_channels=6, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # flat dimensions of the image after conv operations
        fc_input_dim = 16 * self.__calculate_fc_input_dim()

        # Fully connected stack (classifier)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=fc_input_dim, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=84, out_features=num_classes)
        )

    def __calculate_fc_input_dim(self):
        fc_dim = self.input_dim

        # first conv: kernel_size = 5 x 5
        fc_dim = fc_dim - 5 + 1

        # first pool: size = 2 x 2
        fc_dim //= 2

        # second conv: kernel_size = 5 x 5
        fc_dim = fc_dim - 5 + 1

        # second pool: size = 2 x 2
        fc_dim //= 2

        return fc_dim * fc_dim

    def forward(self, x):
        return self.classifier(self.features(x))
