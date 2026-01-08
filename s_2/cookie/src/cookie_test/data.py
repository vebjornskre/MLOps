from __future__ import annotations

import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
from torchvision import transforms
import os

def preprocess_data(raw_path='src/cookie_test/data/raw/', processed_path='src/cookie_test/data/processed/'):
    # exchange with the corrupted mnist dataset
        
    train_images, train_target = [], []
    for i in range(5):
        train_images.append(torch.load(os.path.join(raw_path, f'train_images_{i}.pt')))
        train_target.append(torch.load(os.path.join(raw_path, f'train_target_{i}.pt')))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)
    
    test_images = torch.load(os.path.join(raw_path, 'test_images.pt'))
    test_target = torch.load(os.path.join(raw_path, 'test_target.pt'))

    # We add dimention so it gets 1 channel (grayscale): from (nImages, 28, 28) -> (nImages, 1, 28, 28)
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize images
    mean_tr = train_images.mean()
    std_tr  = train_images.std()
    normalize_tr = transforms.Normalize((mean_tr,), (std_tr,))

    mean_t = test_images.mean()
    std_t  = test_images.std()
    normalize_t = transforms.Normalize((mean_t,), (std_t,))

    train_images_normed = normalize_tr(train_images)
    test_images_normed = normalize_t(test_images)

    torch.save(train_images_normed, os.path.join(processed_path, 'train_images_processed.pt'))
    torch.save(train_target, os.path.join(processed_path, 'train_target_processed.pt'))

    torch.save(test_images_normed, os.path.join(processed_path, 'test_images_processed.pt'))
    torch.save(test_target, os.path.join(processed_path, 'test_target_processed.pt'))


def corrupt_mnist(processed_path='src/cookie_test/data/processed/'):
    """Return train and test dataloaders for corrupt MNIST."""
    try:
        train_images = torch.load(os.path.join(processed_path, 'train_images_processed.pt'))
        train_target = torch.load(os.path.join(processed_path, 'train_target_processed.pt'))

        test_images = torch.load(os.path.join(processed_path, 'test_images_processed.pt'))
        test_target = torch.load(os.path.join(processed_path, 'test_target_processed.pt'))

        train_set = torch.utils.data.TensorDataset(train_images, train_target)
        test_set = torch.utils.data.TensorDataset(test_images, test_target)
    
    except FileNotFoundError:
        print('No processed files in folder... Running preprocess_data() now to generate them')
        print('\nPreprocessing data...')
        preprocess_data(processed_path=processed_path)
        print('\nPreprocessing finished, now creating and returning the datasets...')

        train_images = torch.load(os.path.join(processed_path, 'train_images_processed.pt'))
        train_target = torch.load(os.path.join(processed_path, 'train_target_processed.pt'))

        test_images = torch.load(os.path.join(processed_path, 'test_images_processed.pt'))
        test_target = torch.load(os.path.join(processed_path, 'test_target_processed.pt'))

        train_set = torch.utils.data.TensorDataset(train_images, train_target)
        test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set



def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    print('Preprocessing data if preprocessed_data folder is empty')
    train_set, test_set = corrupt_mnist()
