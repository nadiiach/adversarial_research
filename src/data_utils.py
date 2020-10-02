from torchvision import transforms
import torch.utils.data

from src.dataset import ParallelImageFolders

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
        batch_size=args.batch_size, shuffle=True,
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
        batch_size=args.batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True)

    return train_loader, val_loader
