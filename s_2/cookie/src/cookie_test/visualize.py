import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
import typer

import torch
from torch import nn
from data import corrupt_mnist
from model import MyAwesomeModel

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

app = typer.Typer()

@app.command()
def visualize(
    model_checkpoint: str = typer.Option("models/trained_model.pth", "-f", "--fname"),
    embed: str = typer.Option("f", "-e", "--embed"),
):

    "Plotting some cool visualizations"

    if embed == 't':
        embed = True
    elif embed == 'f':
        embed = False
    else:
        print('Give valid value for -f!!!!!!!!!!!')
    # Load model
    sd = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(sd)

    _, test_set = corrupt_mnist()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    maps = []
    targets = []

    if embed:
        print('Making figure of the embeddings')
        figure_name = 'embed_vis.png'

        embeddings, targets = [], []
        with torch.inference_mode():
            for batch in test_loader:
                images, target = batch
                predictions = model(images, ret_logits=True)
                embeddings.append(predictions)
                targets.append(target)
            embeddings = torch.cat(embeddings).numpy()
            targets = torch.cat(targets).numpy()

        if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
            pca = PCA(n_components=100)
            embeddings = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 10))
        for i in range(10):
            mask = targets == i
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
        plt.legend()
        plt.savefig(f"reports/figures/{figure_name}")
    else:
        print('Making figure of the last conv map')
        figure_name = 'conv_map.png'
        
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(test_loader):
                last_map = model(imgs, ret_last_map=True)

                last_map = last_map.sum(dim=1, keepdim=True)   # → (B, 1, H, W)
                last_map = last_map.sum(dim=0, keepdim=True)   # → (1, H, W)

                maps.append(last_map)
                targets.append(labels)
                if i >= 9:
                    break

        row_col = int(len(maps) ** 0.5)
        fig = plt.figure(figsize=(10.0, 10.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
        for ax, im, label in zip(grid, maps, targets):
            ax.imshow(im.squeeze(), cmap="gray")
            ax.set_title(f"Label: {label.item()}")
            ax.axis("off")
        plt.savefig(f'reports/figures/{figure_name}')
        


if __name__ == '__main__':
    app()