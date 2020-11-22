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
except:
    from generators import GeneratorResnet
    import utils

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', help='comics, imagenet', required=True)
parser.add_argument('--batch_size', type=int, default=15, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--subset', type=int, help='Subset of dataset for testing')
parser.add_argument('--model_type', type=str, default='res152_incv3',
                        help='Model against GAN is trained: vgg16, '
                         'vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img',
                    help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--imgsave', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--foldname',  type=str, required=True,
                    help="In what folder to save trained model in saved_models?")

args = parser.parse_args()
print(args)
if not args.save:
    print("Warning: model will NOT be saved")

# Normalize (0-1)
eps = args.eps / 255
print("eps ", eps)
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
# Model
####################
discriminators_size_larger = {
    "vgg16": torchvision.models.vgg16(pretrained=True),
    # "res152": torchvision.models.resnet152(pretrained=True)
}

discriminators_size_smaller = {
    "incv3": torchvision.models.inception_v3(pretrained=True),
}

for model in list(discriminators_size_larger.values()) \
             + list(discriminators_size_smaller.values()):
    model = model.to(device)
    model.eval()

# Generator
if 'incv3' in args.model_type:
    netG = GeneratorResnet(inception=True)
else:
    netG = GeneratorResnet()

netG.to(device)
# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

img_size_vgg16_19_res152 = 299
img_size_others = 223

# Data
data_transform_vgg16_19_res152 = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(img_size_vgg16_19_res152),
    transforms.ToTensor(),
])

data_transform_others = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size_others),
    transforms.ToTensor(),
])

train_dir = args.train_dir
train_set_vgg16_19_res152 = datasets.ImageFolder(train_dir, data_transform_vgg16_19_res152)

if args.subset:
    train_set_vgg16_19_res152 = torch.utils.data.Subset(
        train_set_vgg16_19_res152, np.random.choice(
            len(train_set_vgg16_19_res152), args.subset, replace=False))
    print("Sampled {} images".format(args.subset))


train_loader_vgg16_19_res152 = torch.utils.data.DataLoader(train_set_vgg16_19_res152,
                                                        batch_size=args.batch_size,
                                                        shuffle=True, num_workers=4,
                                                        pin_memory=True)


train_set_others = datasets.ImageFolder(train_dir, data_transform_others)

if args.subset:
    train_set_others = torch.utils.data.Subset(
        train_set_others, np.random.choice(
            len(train_set_others), args.subset, replace=False))
    print("Sampled {} images".format(args.subset))

train_loader_others = torch.utils.data.DataLoader(train_set_others,
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4,
                                                  pin_memory=True)


if not len(train_loader_vgg16_19_res152) == len(train_loader_others):
    print("len(train_loader_vgg16_19_res152)={} == len(train_set_others)={}".
          format(len(train_loader_vgg16_19_res152), len(train_set_others)))
    exit()
train_size = len(train_loader_vgg16_19_res152)
print('Number of batches:', train_size)
# Loss
criterion = nn.CrossEntropyLoss()
####################
# Set-up noise if required
####################
if args.attack_type == 'noise':
    noise_vgg16_19_res152 = np.random.uniform(0, 1,
                                              img_size_vgg16_19_res152 *
                                              img_size_vgg16_19_res152 * 3)

    noise_others = np.random.uniform(0, 1, img_size_others *
                                     img_size_others * 3)
    # Save noise
    np.save('saved_models/noise_{}_{}_{}_vgg16_19_res152_rl'.format(args.target,
                                                                    args.model_type,
                                                                    args.train_dir),
                                                                    noise_vgg16_19_res152)

    np.save('saved_models/noise_{}_{}_{}_others_rl'.format(args.target,
                                                           args.model_type,
                                                           args.train_dir),
            noise_others)

    im_noise_vgg16_19_res152 = np.reshape(noise_vgg16_19_res152,
                                          (3, img_size_vgg16_19_res152,
                                           img_size_vgg16_19_res152))

    im_noise_others = np.reshape(noise_others, (3, img_size_others,
                                                img_size_others))

    im_noise_vgg16_19_res152 = im_noise_vgg16_19_res152[np.newaxis, :, :, :]
    im_noise_others = im_noise_others[np.newaxis, :, :, :]

    im_noise_vgg16_19_res152_tr = np.tile(im_noise_vgg16_19_res152,
                                          (args.batch_size, 1, 1, 1))
    im_noise_others_tr = np.tile(im_noise_others, (args.batch_size, 1, 1, 1))

    noise_tensor_vgg16_19_res152_tr = torch.from_numpy(
        im_noise_vgg16_19_res152_tr).type(torch.FloatTensor).to(device)

    noise_tensor_others_tr = torch.from_numpy(im_noise_others_tr) \
        .type(torch.FloatTensor).to(device)

# Training
print('Label: {} \t Attack: {} dependent \t Model: {} \t '
      'Distribution: {} \t Saving instances: {}'.format(args.target,
                                                        args.attack_type,
                                                        args.model_type,
                                                        args.train_dir,
                                                        args.epochs))
for epoch in range(args.epochs):
    running_loss = 0
    for i in range(len(train_loader_vgg16_19_res152)):
        (img1, _) = next(iter(train_loader_vgg16_19_res152))
        (img2, _) = next(iter(train_loader_others))
        img1 = img1.to(device)
        img2 = img2.to(device)
        # for (img1, _), (img2, _) in zip(train_loader_vgg16_19_res152,
        #                                              train_loader_others):
        #
        # for data in train_loader_vgg16_19_res152:
        #     img1, img2 = None, None
        clean_labels = []
        for mod in list(discriminators_size_larger.values()) + \
                   list(discriminators_size_smaller.values()):
            clean_labels.append(utils.get_label(img1, device, args, mod))

        netG.train()
        optimG.zero_grad()

        if args.attack_type == 'noise':
            adv1_noise = netG(noise_tensor_vgg16_19_res152_tr)
            adv2_noise = netG(noise_tensor_others_tr)
        else:
            adv1_noise = netG(img1)
            adv2_noise = netG(img2)

        # Projection
        adv1 = utils.projection(adv1_noise, img1, eps)
        adv2 = utils.projection(adv2_noise, img2, eps)

        if args.target == -1:
            # Gradient accent (Untargetted Attack)
            advs_logits_out = []
            imgs_logits_out = []
            for mod in discriminators_size_larger.values():
                advs_logits_out.append(model(utils.normalize(adv1)))
                imgs_logits_out.append(model(utils.normalize(img1)))

            for mod in discriminators_size_smaller.values():
                advs_logits_out.append(model(utils.normalize(adv1)))
                imgs_logits_out.append(model(utils.normalize(img1)))

            # loss = 0
            # for adv_out, img_out, label in zip(advs_logits_out, imgs_logits_out, clean_labels):
            #     loss += -criterion(adv_out - img_out, label)

            losses = []
            for adv_out, img_out, label in zip(advs_logits_out, imgs_logits_out, clean_labels):
                losses.append(-criterion(adv_out - img_out, label))
            loss = min(losses)

        loss.backward()
        optimG.step()

        if i % 10 == 9:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(
                epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())

    discrimins = "_".join(list(discriminators_size_larger.keys()) + \
                          list(discriminators_size_smaller.keys()))


    savestr = 'saved_models/{}/{}_netG_{}_{}_{}_{}_{}_rl.pth'.format(args.foldname, args.foldname,
                                                                     args.target, args.attack_type,
                                                                     discrimins, args.train_dir,
                                                                     epoch)

    if args.save:
        torch.save(netG.state_dict(), savestr)
    else:
        print("Warning: model is not saved!")

    # Save noise
    if args.attack_type == 'noise':
        # Save transformed noise
        t_noise1 = netG(torch.from_numpy(im_noise_vgg16_19_res152_tr)
                        .type(torch.FloatTensor).to(device))
        utils.save_img(t_noise1, args, "vgg16_19_res152", epoch,
                       "noise_pics", "noise")
        t_noise2 = netG(torch.from_numpy(im_noise_others_tr)
                        .type(torch.FloatTensor).to(device))
        utils.save_img(t_noise1, args, "others", epoch,
                       "noise_pics", "noise")

    if args.imgsave is True:
        utils.save_img(adv1_noise, args, "others", epoch, "adv_pics", "noise",
                       save_npy=False)

        utils.save_img(adv1, args, "others", epoch, "adv_pics", "imgwnoise",
                       save_npy=False)
