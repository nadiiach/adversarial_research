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

try:
    from cross_domain_transferable_perturbations.generators import GeneratorResnet
    from cross_domain_transferable_perturbations import utils
    from cross_domain_transferable_perturbations.utils import *
except:
    from generators import GeneratorResnet
    import utils
    from utils import *

PRINT_FREQ = 10

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='imagenet', help='comics, imagenet')
parser.add_argument('--batch_size', type=int, default=15, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--subset', type=int, help='Subset of dataset for testing')
parser.add_argument('--model_type', type=str, default='vgg16',
            help='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img',
            help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--lamb', type=int, default=0.05, help='penalty regularizer')
parser.add_argument('--logdet', action='store_true', help="Van Neumann relaxation")
parser.add_argument('--save', action='store_true', help="save models")
parser.add_argument('--foldname',  type=str, required=True,
                    help="In what folder to save trained model in saved_models?")

args = parser.parse_args()
print(args)

if not args.save:
    print("WARNING: debug mode!!! Models won't be saved.")

if args.batch_size < 8:
    print("WARNING: args.batch_size < 8")

if args.epochs < 10:
    print("WARNING: args.epochs < 10")

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

iteration = -1

for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        iteration += 1

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

        assert len(imgwnoise.shape) == 4
        adv = netG(imgwnoise)

        # Projection
        xx = torch.max(adv, img - eps)
        adv = torch.min(xx, img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        f = adv[:int(img.shape[0]/2)]
        s = adv[int(img.shape[0]/2):]

        assert not torch.equal(f, s)

        if not args.logdet:
            diff = torch.norm(f - s)
        else:
            # convex relaxation of Von Neumann entropy
            final_matrix = torch.zeros((f.shape[-1], f.shape[-1]), dtype=torch.float64).cuda()
            for bat in range(f.shape[0]):
                for ch in range(f.shape[1]):
                    m1 = f[bat][ch]
                    m2 = s[bat][ch]

                    tm = torch.transpose(m1 - m2, 0, 1)
                    m = (m1 - m2) * tm
                    final_matrix = m if final_matrix is None else final_matrix + m

            final_matrix = 1/f.shape[-1] * torch.eye(final_matrix.shape[-1]).cuda()
            diff = torch.logdet(final_matrix)
            diff *= -1
            if torch.isinf(diff) or torch.isnan(diff):
                print(final_matrix)
                print("type ", final_matrix.dtype)
                print("shape ", final_matrix.shape)
                print("norm ",  torch.norm(final_matrix))
                print("determinant ", torch.det(final_matrix))
                print("rank ", torch.matrix_rank(final_matrix))
                print("FATAL: diff = {}".format(diff.item()))
                exit()


        adv_out = model(normalize(adv))
        img_out = model(normalize(img))
        loss = -(criterion(adv_out-img_out, label) + args.lamb * diff)

        loss.backward()
        optimG.step()

        if iteration % 5000 == 0 and iteration != 0:
            if args.save:
                utils.save_snapshot_and_log(netG, args.foldname, args.target,  args.attack_type,
                                    args.train_dir, args.model_type, laststr="{}_iter_{}_bs"
                                    .format(iteration, args.batch_size), iteration=iteration)
            else:
                print("Warning: model is not saved!")

        if iteration % PRINT_FREQ == 0:
            ll = running_loss / 10
            print('Iteration: {0}, Epoch: {1} \t Batch: {2} \t loss: {3:.5f}'.format(
                iteration, epoch, i, running_loss / PRINT_FREQ))
            utils.save_snapshot_and_log(netG, args.foldname, args.target, args.attack_type,
                                        args.train_dir, args.model_type,
                                        st="{} {}".format(iteration, ll), logonly=True,
                                        iteration=iteration)
            running_loss = 0

        running_loss += abs(loss.item())


    if args.save:
        utils.save_snapshot_and_log(netG, args.foldname, args.target,  args.attack_type,
                            args.train_dir, args.model_type, laststr="{}_epoch".format(epoch),
                            iteration=iteration)
    else:
        print("Warning: model is not saved!")

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