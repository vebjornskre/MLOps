from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.cn1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.cn2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.cn3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(25088, 10)


    def forward(self, x):
        x    = F.relu(self.cn1(x))
        x    = F.relu(self.cn2(x))
        x    = F.relu(self.cn3(x))
        # x = self.drop(self.pool(x))
        flat = self.drop(self.flat(x))

        out  = F.softmax(self.out(flat), dim=1)

        return out


if __name__ == '__main__':
    import torch
    print('Imports works')

    torch.manual_seed(7)  # Set the random seed so things are predictable.
    model = MyAwesomeModel()

    # Features are 5 random normal variables.
    test_img = torch.randn((1, 28, 28))
    model(test_img)