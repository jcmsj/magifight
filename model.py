from torch import nn

class HarryNet(nn.Module):
    def __init__(self, num_classes):
        super(HarryNet, self).__init__()
        self.model = nn.Sequential(
            # Expects a grayscale image
            nn.Conv2d(1, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 9, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # for confidence
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

