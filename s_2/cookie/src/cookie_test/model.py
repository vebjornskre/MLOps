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


    def forward(self, x, ret_last_map=False, ret_logits=False):
        x    = F.relu(self.cn1(x))
        x    = F.relu(self.cn2(x))
        x    = F.relu(self.cn3(x))
        if ret_last_map == True:
            return x
        # x = self.drop(self.pool(x))
        flat = self.drop(self.flat(x))
      
        logits = self.out(flat)
        if ret_logits:
            return logits

        out  = F.softmax(self.out(flat), dim=1)

        return out

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print('Model:')
    model = MyAwesomeModel()
    print(model)

    print('\nNumber of model parameter:')
    print(num_params(model))
