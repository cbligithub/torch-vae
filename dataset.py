# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import pathlib

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image

import matplotlib.pyplot as plt


import utils

torch.set_printoptions(threshold=10_000_000, linewidth=1000)
np.set_printoptions(threshold=10_000_000, linewidth=1000)

class HandDatasetSimple(torch.utils.data.Dataset):
    def __init__(self, image_dir=pathlib.Path.home()/"dataset", train=True, transform=transforms.Compose([transforms.ToTensor()])):
        self.image_dir = pathlib.Path(image_dir)
        files = list(self.image_dir.glob("*.png"))
        files.sort()
        # self.image_files = [files[0] for _ in range(400)]
        self.image_files = files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = str(self.image_files[idx])

        # image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # nimg = cv2.resize(image, (64, 48), interpolation=cv2.INTER_NEAREST)
        # image = nimg
        image = np.array(Image.open(img_path))

        nimg = image[:, :, 2].astype(np.uint16) + (image[:, :, 1].astype(np.uint16) << 8)
        nimg = nimg.astype(np.float32)
        nimg = nimg / nimg.max()  # Normalize to [0, 1]
        # nimg = (nimg / 65535.0 * 255.0).astype(np.uint8)

        # plt.figure()
        # plt.imshow(nimg, vmin=0, vmax=np.max(nimg))
        # plt.title("Depth Image")
        # plt.axis('off')
        # plt.show()

        # cv2.imshow("Depth Image", nimg)
        # cv2.waitKey(0)

        image = Image.fromarray(nimg)

        if self.transform:
            image = self.transform(image)
        # print(image.type(), image.shape, image.min(), image.max(), self.transform)
        return image, 0


class HandDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir=pathlib.Path.home()/"dataset", train=True, transform=transforms.Compose([transforms.ToTensor()])):
        if train:
            image_dir = pathlib.Path.home()/"Downloads/nyu_hand_dataset_v2/dataset/train/"
        else:
            image_dir = pathlib.Path.home()/"Downloads/nyu_hand_dataset_v2/dataset/test/"
        self.image_dir = pathlib.Path(image_dir)

        files = list(self.image_dir.glob("depth_1_*.png"))
        files.sort()
        # self.image_files = [files[0] for _ in range(400)]
        self.image_files = files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = str(self.image_files[idx])

        # image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # nimg = cv2.resize(image, (64, 48), interpolation=cv2.INTER_NEAREST)
        # image = nimg
        image = np.array(Image.open(img_path))

        nimg = image[:, :, 2].astype(np.uint16) + (image[:, :, 1].astype(np.uint16) << 8)
        nimg = nimg.astype(np.float32)
        nimg = nimg / nimg.max()  # Normalize to [0, 1]

        image = Image.fromarray(nimg)

        if self.transform:
            image = self.transform(image)
        return image, 0

def get_loaders(dataset_name, img_size, batch_size, root="./data"):
    load_fn = None
    num_img_channels = 0
    if dataset_name == "mnist":
        load_fn = torchvision.datasets.MNIST
        num_img_channels = 1
    elif dataset_name == "fashion-mnist":
        load_fn = torchvision.datasets.FashionMNIST
        num_img_channels = 1
    elif dataset_name == "cifar-10":
        load_fn = torchvision.datasets.CIFAR10
        num_img_channels = 3
    elif dataset_name == "cifar-100":
        load_fn = torchvision.datasets.CIFAR10
        num_img_channels = 3
    elif dataset_name == "celeb-a":
        load_fn = torchvision.datasets.CelebA
        num_img_channels = 3
    elif dataset_name == "hand":
        load_fn = HandDataset
        # load_fn = HandDatasetSimple
        num_img_channels = 1
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_loader, test_loader, classes_list = torchvision_load(
        dataset_name, batch_size, load_fn, img_size, root
    )
    return train_loader, test_loader, classes_list, num_img_channels


def torchvision_load(
    dataset_name, batch_size, load_fn, img_size=(32, 32), root="./data"
):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # resize the image to img_size pixels
            transforms.ToTensor(),  # convert to tensor. This will also normalize pixels to 0-1
        ]
    )

    # load train and test sets using torchvision
    if dataset_name == "celeb-a":
        tr = load_fn(root=root, split="train", download=True, transform=transform)
        test = load_fn(root=root, split="test", download=True, transform=transform)
        classes_list = None  # could use "identity" attribute of the dataset
    elif dataset_name in ["cifar-100", "cifar-10", "mnist", "fashion-mnist"]:
        tr = load_fn(root=root, train=True, download=True, transform=transform)
        test = load_fn(root=root, train=False, download=True, transform=transform)
        classes_list = tr.classes
    elif dataset_name in ["hand"]:
        tr = load_fn(transform=transform, train=True)
        test = load_fn(transform=transform, train=False)
        classes_list = None  # could use "identity" attribute of the dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    return train_loader, test_loader, classes_list


if __name__ == "__main__":
    batch_size = 32
    # img_size = (64, 64)
    img_size = (48*10, 64*10)

    tr_loader, test_loader, classes_list, num_img_channels = get_loaders(
        "hand", img_size=img_size, batch_size=batch_size
    )

    B, C, H, W = batch_size, num_img_channels, img_size[0], img_size[1]
    print(f"Batch size: {B}, Channels: {C}, Height: {H}, Width: {W}")

    images, labels = next(iter(tr_loader))
    assert images.shape == (B, C, H, W), "Wrong training set size"
    assert labels.shape == (B,), "Wrong training set size"

    images, labels = next(iter(test_loader))
    assert images.shape == (B, C, H, W), "Wrong training set size"
    assert labels.shape == (B,), "Wrong training set size"

    print(f"Classes : {classes_list}")

    # Save an image as a sanity check

    # Convert PyTorch tensor to numpy array and scale to 0-255
    # img_data = (images[10].detach().cpu().numpy() * 255).astype(np.uint8)
    # print(images[0].detach().cpu().numpy().dtype)
    img = images[0].detach().cpu().numpy()
    print(img.shape, img.dtype, np.min(img), np.max(img))
    img_data = (img * 255).astype(np.uint8)

    # Save the image using Pillow
    utils.save_image(img_data, "/tmp/trainTmp.png")

    print("Dataset prepared successfully!")
