from torchvision import transforms
import torch.utils.data

from src.dataset import ParallelImageFolders

DATASET_PATHS = {
    'places': '/data/vision/torralba/datasets/places/files',
    'imagenet': '/data/vision/torralba/datasets/imagenet_pytorch'
}

# Dictionary of dataset norms (what is the norm of the places dataset, shouldn't we be normalizing to that?)
NORMALIZER = {
    'imagenet': transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
}


def get_loaders(training_dir, val_dir, batch_size, resize=256, crop=224, train_workers=48, val_workers=24):
    train_loader = torch.utils.data.DataLoader(
        ParallelImageFolders([training_dir],
                             classification=True,
                             transform=transforms.Compose([
                                 transforms.Resize(resize),
                                 transforms.RandomCrop(crop),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 NORMALIZER['imagenet'],
                             ])),
        batch_size=batch_size, shuffle=True,
        num_workers=train_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ParallelImageFolders([val_dir],
                             classification=True,
                             transform=transforms.Compose([
                                 transforms.Resize(resize),
                                 transforms.CenterCrop(crop),
                                 transforms.ToTensor(),
                                 NORMALIZER['imagenet'],
                             ])),
        # batch_size=64, shuffle=False,
        batch_size=batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True)

    return train_loader, val_loader


def get_loader(val_dir, batch_size, resize=256, crop=224, train_workers=48, val_workers=24):
    return torch.utils.data.DataLoader(
        ParallelImageFolders([val_dir],
                             classification=True,
                             transform=transforms.Compose([
                                 transforms.Resize(resize),
                                 transforms.CenterCrop(crop),
                                 transforms.ToTensor(),
                             ])),
        # batch_size=64, shuffle=False,
        batch_size=batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True)


import src.batch_transforms


def get_normalization_transforms(device='cpu'):
    """Transforms image to ImageNet mean/variance. Also returns reverse transform"""
    forward = transforms.Compose([  # [1]
        src.batch_transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225],  # [7]
            device=device
        )])
    reverse = transforms.Compose([
        src.batch_transforms.Normalize(mean=[0., 0., 0.],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                                       device=device),
        src.batch_transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                       std=[1., 1., 1.],
                                       device=device),
    ])
    return forward, reverse
