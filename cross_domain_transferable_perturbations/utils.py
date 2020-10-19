import torch
import torchvision
from cross_domain_transferable_perturbations.generators import GeneratorResnet
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load a particular generator
def load_gan(args):
    # Load Generator
    if args.model_type == 'incv3':
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()

    print('Label: {} \t Attack: {} dependent \t Model: {} '
          '\t Distribution: {} \t Saving instance: {}'.format(args.target,
                                                               args.attack_type,
                                                               args.model_type,
                                                               args.train_dir,
                                                               args.epochs))
    if args.rl:
        netG.load_state_dict(
            torch.load(
                'cross_domain_transferable_perturbations/'
                'saved_models/netG_{}_{}_{}_{}_{}_rl.pth'.format(args.target,
                                                             args.attack_type,
                                                             args.model_type,
                                                             args.train_dir,
                                                             args.epochs)))
    else:
        netG.load_state_dict(
            torch.load(
                'cross_domain_transferable_perturbations/'
                'saved_models/netG_{}_{}_{}_{}_{}.pth'.format(args.target,
                                                          args.attack_type,
                                                          args.model_type,
                                                          args.train_dir,
                                                          args.epochs)))
    return netG


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

def save_img(img_tensor, args, suffix, epoch):
    t_noise_np = np.transpose(img_tensor[0].detach().cpu().numpy(), (1, 2, 0))
    f = plt.figure()
    plt.imshow(t_noise_np, interpolation='spline16')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    f.savefig('saved_models/noise_transformed_{}_{}_{}_{}_{}_rl'.
              format(args.target, args.model_type,
                     args.train_dir, epoch, suffix)
              + ".pdf", bbox_inches='tight')
    np.save('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(
        args.target, args.model_type, args.train_dir, epoch), t_noise_np)