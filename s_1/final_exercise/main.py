import torch
from torch import nn
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, e=5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, test_set = corrupt_mnist()

    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    statistics = {"train_loss": [], "train_accuracy": [], 'test_loss': [], 'test_accuracy':[]}

    for i in range(int(e)):
        run_train_loss = 0
        run_train_acc = 0
        
        run_test_loss = 0
        run_test_acc = 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()

            out = model(imgs)

            loss = lossfn(out, labels)

            loss.backward()
            optimizer.step()

            run_train_loss += loss.item()
            
            accuracy = (out.argmax(dim=1) == labels).float().mean().item()
            run_train_acc += accuracy

        statistics["train_loss"].append(run_train_loss/len(train_loader))
        statistics["train_accuracy"].append(run_train_acc/len(train_loader))

        with torch.no_grad():
            for imgs, labels in test_loader:
                out = model(imgs)
                loss = lossfn(out, labels)
                
                accuracy = (out.argmax(dim=1) == labels).float().mean().item()
                
                run_test_loss += loss.item()
                run_test_acc += accuracy

            statistics["test_loss"].append(run_test_loss/len(test_loader))
            statistics["test_accuracy"].append(run_test_acc/len(test_loader))
        # for plotting loss curve
        print(f'Loss in epoch {i}: {run_train_loss/len(train_set)}')

    torch.save(model.state_dict(),'trained_model.pth')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], '-o')
    axs[0].plot(statistics['test_loss'], '-o')
    axs[0].legend(["train_loss", "test_loss"])
    axs[0].set_title("Train and Test loss")

    axs[1].plot(statistics["train_accuracy"], '-o')
    axs[1].plot(statistics["test_accuracy"], '-o')
    axs[1].legend(["train_accuracy", "test_accuracy"])
    axs[1].set_title("Train and Test accuracy")

    fig.savefig("training_statistics.png")

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


if __name__ == "__main__":
    app()
