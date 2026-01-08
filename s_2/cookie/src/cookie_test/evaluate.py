import torch
from torch import nn
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

app = typer.Typer()

@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    corr = 0
    tot = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs)
            corr += (out.argmax(dim=1) == labels).float().sum().item()
            tot += labels.size(0)

    print(f"Test accuracy: {corr / tot}")

if __name__ == '__main__':
    app()