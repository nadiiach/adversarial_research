import torch
import torchvision
import os
try:
    from cross_domain_transferable_perturbations.generators import GeneratorResnet
except:
    from generators import GeneratorResnet

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from datetime import datetime

now = datetime.now() # current date and time

# Load a particular generator

def _get_path(target, attack_type, model_type, epochs, prefix, ds):
    if args.rl:
        path = 'saved_models/{}/{}_netG_{}_{}_{}_{}_{}_epoch_rl.pth'.format(
                    prefix, prefix, target, attack_type, model_type, ds, epochs)
    else:
        path = 'saved_models/{}/{}_netG_{}_{}_{}_{}_{}_epoch.pth'.format(
                    prefix, prefix, target, attack_type, model_type, ds, epochs)
    return path


def testload_gan():
    pass

def load_gan(model_type, data_dir, target, attack_type, epochs, pth_name, prefix=None, path=None, channels=3):
    assert prefix or path 
    # Load Generator
    if model_type == 'incv3':
        netG = GeneratorResnet(inception=True, channels=channels)
    else:
        netG = GeneratorResnet(channels=channels)

    if "imagenet" in data_dir:
        ds = "imagenet"

    print('Label: {} \t Attack: {} dependent \t Model: {} '
          '\t Distribution: {} \t Saving instance: {}'.format(target, attack_type, 
                                                              model_type, ds, epochs))
    if not path:
        path = _get_path(target, attack_type, model_type, epochs, prefix, ds) \
            if not pth_name \
            else os.path.join("saved_models/{}".format(prefix), pth_name)
    netG.load_state_dict(torch.load(path))

    print("Loaded {}".format(path))
    return netG

#saved_models/test_frobdiff/test_frobdiff_netG_-1_img_vgg16_imagenet_0_epoch_rl.pth
#saved_models/test_frobdiff/test_frobdiff_netG_-1_img_vgg16_imagenet_0_epoch_rl.pth
# Load ImageNet model to evaluate
def load_model(args):
    # Load Targeted Model
    if args.model_t == 'dense201':
        model_t = torchvision.models.densenet201(pretrained=True)
    elif args.model_t == 'vgg19':
        model_t = torchvision.models.vgg19(pretrained=True)
    elif args.model_t == 'vgg16':
        model_t = torchvision.models.vgg16(pretrained=True)
    elif args.model_t == 'incv3':
        model_t = torchvision.models.inception_v3(pretrained=True)
    elif args.model_t == 'res152':
        model_t = torchvision.models.resnet152(pretrained=True)
    elif args.model_t == 'res50':
        model_t = torchvision.models.resnet50(pretrained=True)
    elif args.model_t == 'sqz':
        model_t = torchvision.models.squeezenet1_1(pretrained=True)
    return model_t

############################################################
# If you have all 1000 class folders. Then using default loader is ok.
# In case you have few classes (let's 50) or collected random images in a folder
# then we need to fix the labels.
# The code below will fix the labels for you as long
# as you don't change "orginal imagenet ids".
# for example "ILSVRC2012_val_00019972.JPEG ... "

def fix_labels(test_set, val_txt_path):
    val_dict = {}
    with open(val_txt_path) as file:
        for line in file:
            (key, val) = line.split(' ')
            val_dict[key.split('.')[0]] = int(val.strip())

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[
            test_set.samples[i][0].split('/')[-1].split('.')[0]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
############################################################


#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(test_set, pytorch=False):

    '''
    :param pytorch: pytorch models have 1000 labels as compared to
    tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv("images.csv")
    image_metadata = pd.DataFrame({"ImageId": [f[:-4]
                        for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()

    # Populate the dictionary: key(image path),
    # value ([true label, target label])
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label-1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples

    return test_set


# Rescale image b/w (-1, +1)
def rescale(image):
    return image*2-1

def projection(adv, img, eps):
    xx = torch.max(adv, img - eps)
    adv = torch.min(xx, img + eps)
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv

def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

def get_label(img, device, args, model):
    img = img.to(device)

    if args.target == -1:
        # whatever the model think about the input
        inp = normalize(img.clone().detach())
        label = model(inp).argmax(dim=-1).detach()
    else:
        label = torch.LongTensor(img.size(0))
        label.fill_(args.target)
        label = label.to(device)
    return label

def save_img(img_tensor, args, suffix, epoch, folder, name, save_npy=True):
    img = np.transpose(img_tensor[0].detach().cpu().numpy(), (1, 2, 0))
    img = np.float32(img)
    img = cv2.normalize(img, None, alpha=0,
                        beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    f = plt.figure()
    plt.imshow(img, interpolation='spline16')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    print("Saving noise pictures after epoch {}".format(epoch))

    figpath = '/mnt/Vol2TBSabrentRoc/Projects/adversarial_research/' \
                'cross_domain_transferable_perturbations/{}/' \
                '{}_{}_{}_{}_{}_{}_rl.pdf'\
                .format(folder, name, args.target, args.model_type,
                        args.train_dir, epoch, suffix)
    npypath = '/mnt/Vol2TBSabrentRoc/Projects/adversarial_research/' \
                'cross_domain_transferable_perturbations/{}/' \
                '{}_{}_{}_{}_{}_rl.npy'\
                .format(folder, name, args.target, args.model_type,
                        args.train_dir, epoch)

    f.savefig(figpath, bbox_inches='tight')
    if save_npy:
        np.save(npypath, img)

    print("Saved {}".format(figpath))
    print("Saved {}".format(npypath))

def img_noise(img_size, args, device, batch_size, channels=1):
    noise = np.random.uniform(0, 1, img_size * img_size * channels)
    # Save noise
    # np.save('saved_models/noise_{}_{}_{}_rl'.format(args.target,
    #                                                 args.model_type,
    #                                                 args.train_dir), noise)
    im_noise = np.reshape(noise, (channels, img_size, img_size))
    im_noise = im_noise[np.newaxis, :, :, :]
    im_noise_tr = np.tile(im_noise, (batch_size, 1, 1, 1))
    noise_tensor_tr = torch.from_numpy(im_noise_tr).\
                type(torch.FloatTensor).to(device)
    return noise_tensor_tr


def save_snapshot_and_log(netG, foldname, target, attack_type,
                          train_dir, model_name, st="", verbose=False, logonly=False, iteration=-1):

    pp = 'saved_models/{}'.format(foldname)

    if iteration == 0 and os.path.exists(pp):
        timestr = now.strftime("%d_%m-%H_%M-%S")
        os.rename(pp, pp + "_" + timestr + "_bak")
        assert not os.path.exists(pp)

    if not os.path.exists(pp):
        os.mkdir(pp)

    ds = train_dir.split("/")[-1]

    if "imagenet" in ds:
        ds = "imagenet"

    if logonly:
        logstr = 'saved_models/{}/{}_netG_{}_{}_{}_{}_rl.log'.format(foldname, foldname, target,
                                                                     attack_type, model_name, ds)

        if iteration == 0 and os.path.exists(logstr):
            os.remove(logstr)
            print("Removed old log file at ", logstr)

        with open(logstr, "a+") as f:
            f.write(st + '\n')

        return

    savestr = 'saved_models/{}/{}_netG_{}_{}_{}_{}_{}_rl.pth'.format(foldname, foldname, target,
                                                                     attack_type, model_name,
                                                                     ds, st)
    torch.save(netG.state_dict(), savestr)
    if not verbose:
        print("Model saved to {}".format(savestr))

