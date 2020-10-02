import os
from adversarial_attacks import test
from utils import util_functions as uf
import matplotlib.pyplot as plt
import torch
import cv2

train_len = input("short or long:").strip()
iter_prefix = "iter"  # "best_iter"
pos = len(iter_prefix.split("_"))

def list_dirs(model):
    p = "checkpoints/{}_trained/{}".format(train_len, model)
    ll = [x for x in os.listdir(p) if iter_prefix in x]
    ll = [int(x.split("_")[pos]) for x in ll]
    ll = sorted(ll)
    return ll

def get_image_both_sizes():
    image_batch, labels, orig_paths, names, \
    cats, orig_images, catlist = test.get_batch("resnet18")
    path = orig_paths[0][0]
    image_batch_alex, labels_alex, paths = \
        uf.get_image_as_batch(path=path, val_data_folder="datasets/places/val",
                              val=True, model="alexnet")
    image_batch_alex, labels_alex = image_batch_alex.cuda(), labels_alex.cuda()
    assert torch.equal(labels, labels_alex)
    return image_batch_alex, image_batch, labels, orig_paths

#
# ll1 = list_dirs("softresnet18")
# ll2 = list_dirs("softalexnet")
# ll3 = list_dirs("alexnet")
# ll4 = list_dirs("resnet18")
# print(ll3)
# print(ll4)
# mi = min([
#     # ll1[-1],
#     # ll2[-1],
#     ll3[-1],
#     ll4[-1]
# ])
# print(mi)

#models = ["alexnet", "resnet", "softalexnet", "softresnet"]
models = ["alexnet"]

if train_len == "long":
    iters = [131072]
else:
    iters = [131072]

epsilons = [0.0001, 0.001, 0.01, 0.1]

image_batch_alex, image_batch, labels, orig_paths = None, None, None, None
attempt = 0

#find image that is classified correctly for all models and iters
while attempt < 50:
    print("attempt = {}".format(attempt))
    image_batch_alex, image_batch, labels, orig_paths = get_image_both_sizes()
    print(orig_paths[0][0])
    attempt += 1

    failed = False
    for mod in models:
        for it in iters:
            model = uf.get_model_arg(mod, it, longtrain=train_len=="long")
            model.eval()
            model.cuda()

            if "alex" in mod:
                out = model(image_batch_alex.cuda())
            else:
                out = model(image_batch.cuda())

            predcat = torch.argmax(out).cpu().numpy()
            label = labels.cpu().numpy()[0]
            if predcat != label:
                print("predcat={} != labels={}, model={}, it={}".format(
                    predcat, label, mod, it))
                failed = True
                break
        if failed:
            break

if failed:
    print("Didn't find an image with correct classification!")
    exit()
rows = 1
cols = 2
figsize = (cols * 8, rows * 5)
fig, axs = plt.subplots(rows, cols, figsize=figsize)
mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                    wspace=None, hspace=None)
plt.suptitle(orig_paths[0][0], x=mid, y=1)

for i, mod in enumerate(models):
    for it in iters:
        ax = axs[i]
        epsilons, robust_accuracy, raw, clipped, \
            is_adv, (predcat, adv_predcat, label) = test.run_attacks(epsilons, it,
                image_batch if "alexnet" not in mod else image_batch_alex,
                    labels, attack_name="PGD", longtrain=train_len=="long")

        ax.title("{}".format(mod))
        ax.plot(iters, robust_accuracy, label="it {}".format(it))

#uf.verify_same_imgs(image_batch, image_batch_alex, title=path)

