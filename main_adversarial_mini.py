import os
from adversarial_attacks.test import get_batch
from utils import util_functions as uf
import matplotlib.pyplot as plt
import torch

def list_dirs(model):
    p = "checkpoints/long_train/{}".format(model)
    ll = [x for x in os.listdir(p) if "best_iter" in x]
    ll = [int(x.split("_")[2]) for x in ll]
    ll = sorted(ll)
    return ll


ll1 = list_dirs("softresnet18")
ll2 = list_dirs("softalexnet")
ll3 = list_dirs("alexnet")
ll4 = list_dirs("resnet18")

mi = min([ll1[-1], ll2[-1], ll3[-1], ll4[-1]])
print(mi)

iters = [16284, 19384, 23170, 27454, 32768, 38868, 46241, 55009, 92582, 92682, 110118, 110218]

image_batch, labels, orig_paths, names, cats, orig_images, catlist = get_batch("resnet18")
path = orig_paths[0][0]
print(path)


image_batch_alex, labels_alex, orig_paths, names, cats, orig_images, catlist = \
    uf.get_image_as_batch(path=image_batch, val=True, model="alexnet")