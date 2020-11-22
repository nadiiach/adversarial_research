import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from cross_domain_transferable_perturbations.generators import GeneratorResnet
from cross_domain_transferable_perturbations.utils import *


parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='imagenet', help='comics, imagenet')
parser.add_argument('--batch_size', type=int, default=8, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--subset', type=int, help='Subset of dataset for testing')
parser.add_argument('--model_type', type=str, default='vgg16',
            help='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img',
            help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--logdet', action='store_true', help="Van Neumann relaxation")
parser.add_argument('--save', action='store_true', help="save models")
parser.add_argument('--foldname',  type=str, required=True,
                    help="In what folder to save trained model in saved_models?")

args = parser.parse_args()
print(args)

if not args.save:
    print("WARNING: debug mode!!! Models won't be saved.")

# Normalize (0-1)
eps = args.eps/255
print("eps ", eps)
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
# Model
####################
if args.model_type == 'vgg16':
    model = torchvision.models.vgg16(pretrained=True)
elif args.model_type == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
elif args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res152':
    model = torchvision.models.resnet152(pretrained=True)
model = model.to(device)
model.eval()

# Input dimensions
if args.model_type in ['vgg16', 'vgg19', 'res152']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299

channels = 4
# Generator
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True, channels=channels)
else:
    netG = GeneratorResnet(channels=channels)
netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t

train_dir = args.train_dir
train_set = datasets.ImageFolder(train_dir, data_transform)

if args.subset:
    train_set = torch.utils.data.Subset(
        train_set, np.random.choice(
            len(train_set), args.subset, replace=False))
    print("Sampled {} images".format(args.subset))

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=4,
                                           pin_memory=True)
train_size = len(train_set)
print('Number of batches:', train_size)

# Loss
criterion = nn.CrossEntropyLoss()

# Training
print('Label: {} \t Attack: {} dependent \t Model: {} \t '
      'Distribution: {} \t Saving instances: {}'.format(args.target,
                                                        args.attack_type,
                                                        args.model_type,
                                                        args.train_dir,
                                                        args.epochs))
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        img = torch.cat([img] * 2)
        s = img.shape
        noise = torch.rand((s[0], 1, s[2], s[3]))
        assert not torch.all(noise[0].eq(noise[1]))
        imgwnoise = torch.cat((img, noise), 1)
        imgwnoise = imgwnoise.to(device)
        img = img.to(device)

        if args.target == -1:
            # whatever the model think about the input
            inp = normalize(img.clone().detach())
            label = model(inp).argmax(dim=-1).detach()
        else:
            label = torch.LongTensor(img.size(0))
            label.fill_(args.target)
            label = label.to(device)

        netG.train()
        optimG.zero_grad()
        #print(imgwnoise.shape)
        adv = netG(imgwnoise)

        # print(noise.shape)
        # print(adv.shape)
        # print(img.shape)
        # exit()
        # Projection
        xx = torch.max(adv, img - eps)
        adv = torch.min(xx, img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        f = adv[:int(img.shape[0]/2)]
        s = adv[int(img.shape[0]/2):]

        if not args.logdet:
            diff = torch.norm(f - s)
        else:
            # for each img compute
            # (f[i] - s[i]) * (f[i] - s[i])^T
            # sum over all i logdet (f[i] - s[i]) * (f[i] - s[i])^T   try add or subtract...
            # convex relaxation of Von Neumann entropy
            # Try Wassertain

            pass

        adv_out = model(normalize(adv))
        img_out = model(normalize(img))
        loss = -(criterion(adv_out-img_out, label) + 0.05 * diff)

        loss.backward()
        optimG.step()

        if i % 10 == 9:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(
                epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())

    if args.save:
        torch.save(netG.state_dict(), 'saved_models/div_netG_{}_{}_{}_{}_{}_rl.pth'
                   .format(args.target, args.attack_type, args.model_type,
                           args.train_dir, epoch))
    else:
        print("Warning: model is not saved! save flag wasn't turned on.")

    # Save noise
    if args.attack_type == 'noise':
        # Save transformed noise
        t_noise = netG(torch.from_numpy(noise)
                       .type(torch.FloatTensor).to(device))
        t_noise_np = np.transpose(t_noise[0].detach().cpu().numpy(), (1,2,0))
        f = plt.figure()
        plt.imshow(t_noise_np, interpolation='spline16')
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        f.savefig('saved_models/noise_transformed_{}_{}_{}_{}_rl'.
                  format(args.target, args.model_type, args.train_dir, epoch)
                            + ".pdf", bbox_inches='tight')
        np.save('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(
            args.target, args.model_type, args.train_dir, epoch), t_noise_np)