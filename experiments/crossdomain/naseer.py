"""
Code for cross-domain adversarial robustness (modeled after https://arxiv.org/abs/1905.11736)
"""
import argparse

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

import sys

sys.path.append('../..')

from models.resnet_gen import GeneratorResnet
from src import get_normalization_transforms, get_loader

parser = argparse.ArgumentParser(prog="Benchmark Crossdomain Attack", description="")
parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int)
parser.add_argument('-n', '--epochs', default=40, dest='epochs', type=int)
parser.add_argument('-b', '--batch_size', default=16, dest='batch_size', type=int)
parser.add_argument('--lr', default=0.0002, dest='lr', type=float)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')


def train_attack(data_loader, discriminator, normalize, eps=10 / 255):
    generator = GeneratorResnet().to(device)
    discriminator = discriminator.to(device)
    discriminator.eval()

    optim = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (img, _, _) in progress_bar:
            img = img.to(device)

            adv = generator(img)
            adv = torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)  # project

            adv_out = discriminator(normalize(adv))
            img_out = discriminator(normalize(img))
            loss = -F.cross_entropy(adv_out - img_out, img_out.argmax(dim=-1).detach())

            loss.backward()
            optim.step()

            progress_bar.set_postfix(loss=loss.item(), epoch=epoch)

        torch.save(generator.state_dict(), f'saves/epoch{epoch}.pth')


class DenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    loader = get_loader('/data/vision/torralba/datasets/chexnet', batch_size=args.batch_size)
    normalize, reverse_normalize = get_normalization_transforms(device=device)

    model = DenseNet121(pretrained=False, num_classes=14).to(device)
    model_path = '/data/vision/torralba/scratch/raunakc/pytorch_models/m-25012018-123527.pth.tar'
    state_dict = torch.load(model_path)['state_dict']
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        name = name.replace('.1.', '1.').replace('.2.', '2.')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    train_attack(loader, model, normalize)