from torchvision import transforms
from utils import parallelfolder
import torch
import pathlib
import os
from os.path import join, exists
from models import oldvgg16
from models.oldalexnet import AlexNet, AlexNetH
from models.softalexnet import AlexNet as SoftAlexNet
from models.softalexnet import AlexNetH as SoftAlexNetH
from models import softresnet, softvgg
import torchvision
import matplotlib.pyplot as plt

def check_if_item_present(arr, item):
    for i in arr:
        if i == item:
            return True
    return False


def find_child_dir_path(dir_name, max_depth=5, logger=None):
    found = False
    path = pathlib.Path()

    if logger is not None:
        logger.debug("Root search path {}".format(path))

    while not found or max_depth == 0:
        if check_if_item_present(os.listdir(path.absolute()), dir_name):
            found = True
            break
        path = path.parent()
        max_depth -= 1
    path = path.absolute()

    if not found:
        raise Exception("Folder {} was not found!".format(dir_name))

    p = join(path, dir_name)
    return p


def get_available_iters_and_weights_paths(model, checkpoints_directory="checkpoints", logger=None):
    p = find_child_dir_path(checkpoints_directory, logger=logger)
    p = join(p, model)

    if logger is not None:
        logger.debug("Collecting iters and weights paths in {}".format(p))

    if not exists(p):
        raise Exception("Path {} does not exist".format(p))

    files = os.listdir(p)
    weights = [x for x in files if ".pth" in x]
    dic = {}

    for w in weights:
        if "best" in w:
            continue
        try:
            it = int(w.split("_")[1])
        except Exception as e:
            if logger is not None:
                logger.error("Exception {}, w={}".format(e, w))
            continue
        dic[it] = join(p, w)

    return dic


def get_places_transformer(crop_size,
                           dataset_mean=(0.485, 0.456, 0.406),
                           dataset_std=(0.229, 0.224, 0.225),
                           logger=None):
    assert crop_size is not None

    if logger is not None:
        logger.debug("Creating transformer with crop_size={}, "
                     "mean={}, std={}".format(crop_size, dataset_mean, dataset_std))

    trans_list = [
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ]
    transform = transforms.Compose(trans_list)
    return transform

def get_places_normalized_transformer(crop_size,
                           dataset_mean=(0.485, 0.456, 0.406),
                           dataset_std=(0.229, 0.224, 0.225),
                           logger=None):
    assert crop_size is not None

    if logger is not None:
        logger.debug("Creating transformer with crop_size={}, "
                     "mean={}, std={}".format(crop_size, dataset_mean, dataset_std))

    trans_list = [
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x/255),
        #transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ]
    transform = transforms.Compose(trans_list)
    return transform


def load_dataset(data_dir_path, crop_size, paths=(),
                 classification=True, shuffle=True,
                 logger=None,
                 dataset_mean=(0.485, 0.456, 0.406),
                 dataset_std=(0.229, 0.224, 0.225), normalized_version=False):

    if normalized_version:
        g_places_transform = get_places_normalized_transformer(crop_size, dataset_mean, dataset_std, logger=logger)
    else:
        g_places_transform = get_places_transformer(crop_size, dataset_mean, dataset_std, logger=logger)

    if logger is not None:
        logger.debug("Loading dataset from folder {}".format(data_dir_path))

    return parallelfolder.ParallelImageFolders([data_dir_path],
                                               classification=classification,
                                               shuffle=shuffle,
                                               transform=g_places_transform,
                                               paths=paths
                                               )


def get_dataloader(data_dir_path, crop_size,
                   batch_size, num_workers, shuffle=True, pin_memory=True,
                   logger=None,
                   dataset_mean=(0.485, 0.456, 0.406),
                   dataset_std=(0.229, 0.224, 0.225),
                   normalized_version=False):

    if logger is not None:
        logger.debug("Data dir path {}".format(data_dir_path))
        logger.debug("Creating dataset with batch_size {}, "
                     "num_workers={}".format(batch_size, num_workers))

    pig = load_dataset(data_dir_path, crop_size,
                       dataset_mean=dataset_mean,
                       dataset_std=dataset_std,
                       logger=logger, normalized_version=normalized_version)

    dl = torch.utils.data.DataLoader(pig,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)
    return dl


def denorm(img_input,
           dataset_mean=(0.485, 0.456, 0.406),
           dataset_std=(0.229, 0.224, 0.225)):

    if len(img_input.shape) == 4:
        x = img_input.new(*img_input.size())
        x[:, 0, :, :] = img_input[:, 0, :, :] * dataset_std[0] + dataset_mean[0]
        x[:, 1, :, :] = img_input[:, 1, :, :] * dataset_std[1] + dataset_mean[1]
        x[:, 2, :, :] = img_input[:, 2, :, :] * dataset_std[2] + dataset_mean[2]
    else:
        x = img_input.new(*img_input.size())
        x[0, :, :] = img_input[0, :, :] * dataset_std[0] + dataset_mean[0]
        x[1, :, :] = img_input[1, :, :] * dataset_std[1] + dataset_mean[1]
        x[2, :, :] = img_input[2, :, :] * dataset_std[2] + dataset_mean[2]
    return x


def get_image_from_tensor(image):
    image = denorm(image)
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0).numpy()
    if len(image.shape) == 4:
        image = image.permute(0, 2, 3, 1).numpy()[0]
    return image


def get_name_from_path(paths):
    if not isinstance(paths, list):
        paths = [paths]
    names = []
    cats = []
    for p in paths:
        arr = p.split("/")[-2:]
        names.append(arr[1])
        cats.append(arr[0])

    out_name = names if len(names) > 1 else names[0]
    out_cats = cats if len(cats) > 1 else cats[0]
    return out_name, out_cats


def get_batch_of_images(loader):
    assert loader is not None
    image_batch, labels, paths = next(iter(loader))
    name_arr = []
    cat_arr = []
    for p in paths:
        name, cat = get_name_from_path(p)
        name_arr.append(name)
        cat_arr.append(cat)
    orig_images = []
    for img in image_batch:
        orig_images.append(get_image_from_tensor(img))
    return image_batch, labels, paths, name_arr, cat_arr, orig_images

def get_model_arg(model_name, it, longtrain=True, logger=None):
    cpname = "iter_{}_weights.pth".format(it)
    root = os.getcwd()
    cp = join(root, "checkpoints", "long_trained" if longtrain
              else "short_trained",  model_name, cpname)

    if logger is not None:
        logger.debug("Loading checkpoint {}".format(cp))

    checkpoint = torch.load(cp)

    if model_name == "vgg16":
        model = oldvgg16.vgg16(num_classes=365)  # load_original_vgg16()
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == "softvgg16":
        model = softvgg.vgg16(num_classes=365)  # load_original_vgg16()
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == "alexnet" or model_name == "alexnet-long":
        model = AlexNet(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])
        model = AlexNetH(model)

    elif model_name == "softalexnet":
        model = SoftAlexNet(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])
        model = SoftAlexNetH(model)

    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == 'softresnet50':
        model = softresnet.resnet18(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == 'resnet18' or model_name == 'resnet18-forever' \
            or model_name == 'resnet18-long':
        model = torchvision.models.resnet18(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == 'softresnet18':
        model = softresnet.resnet18(num_classes=365)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        raise Exception("Wrong model name {}".format(model_name))
    return model


def _show(image, name):
    plt.imshow(image)
    plt.axis('off')
    plt.title(name)
    plt.savefig(name)
    plt.show()


def get_image_from_tensor(image):
    image = denorm(image)
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0).cpu().numpy()
    if len(image.shape) == 4:
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    return image


def imshow(image, name=""):
    image = get_image_from_tensor(image)
    _show(image, name)

"""get specific image returned in a batch format"""
def get_image_as_batch(clas="", name="", path="",
                       train_data_fold="datasets/places/train",
                       val_data_folder="datasets/places/val",
                       val=False, logger=None, crop_size=-1, model=None):

    if crop_size == -1 and model is not None:
        crop_size = 227 if "alexnet" in model else 224
    elif crop_size == -1 and model is None:
        crop_size = 227

    if len(path) == 0:
        path = join(train_data_fold if not val else val_data_folder, clas, name)

    if logger is not None:
        logger.debug("Loading dataset in get_image_as_batch ...")

    folder = train_data_fold if not val else val_data_folder
    pig = load_dataset(folder, crop_size, paths=(path))

    if logger is not None:
        logger.debug("Finished loading dataset in get_image_as_batch ...")

    dl = torch.utils.data.DataLoader(pig, batch_size=1, shuffle=True,
                                     num_workers=0, pin_memory=True)
    image_batch, labels, paths = next(iter(dl))
    return image_batch, labels, paths


'''
hints

alexnet cropsize 227
resnet, vgg cropsize 224
'''