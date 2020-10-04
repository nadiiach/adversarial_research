import os
from adversarial_attacks import test
from utils import util_functions as uf
import matplotlib.pyplot as plt
import torch
import argparse
import cv2
models = ["alexnet", "resnet18", "softalexnet", "softresnet18"]
# [32768, 65536, 131072]
iters = [16284, 19384, 23170, 27454, 32768, 38868, 46241,
                55009, 92582, 92682, 110118, 110218]
epsilons = [0.0001, 0.001, 0.01, 0.1]

parser = argparse.ArgumentParser()
parser.add_argument('--len', type=str, default="long")
parser.add_argument('--iters', nargs='+', default=iters)
parser.add_argument('--models', nargs='+', default=models)
parser.add_argument('--eps', nargs='+', default=epsilons)
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--minex', type=int, default=65)
parser.add_argument('--cpefix', type=str, choices=["best_iter", "iter"],
                                    default="best_iter")

args = parser.parse_args()
iters = args.iters
BATCH_SIZE = args.batch_size
MIN_EXAMPLES = args.minex
train_len = args.len
iter_prefix = args.cpefix  # "best_iter"

pos = len(iter_prefix.split("_"))


def list_dirs(_model):
    p = "checkpoints/{}_trained/{}".format(train_len, _model)
    ll = [x for x in os.listdir(p) if iter_prefix in x]
    ll = [int(x.split("_")[pos]) for x in ll]
    ll = sorted(ll)
    return ll


def get_image_both_sizes(dataset="places", val=True):
    image_batch, labels, orig_paths, names, \
    cats, orig_images, catlist = test.get_batch("resnet18",
                                                batch_size=BATCH_SIZE)
    paths = orig_paths[0]
    base = uf.find_child_dir_path("datasets")
    basepath_val = os.path.join(base, dataset, "val")
    basepath_train = os.path.join(base, dataset, "train")
    image_batch_alex, labels_alex, paths = \
        uf.get_image_as_batch(paths=tuple(paths),
                              val_data_folder=basepath_val,
                              train_data_fold=basepath_train,
                              val=val, model="alexnet",
                              batch_size=BATCH_SIZE)
    image_batch_alex, labels_alex = image_batch_alex.cuda(), labels_alex.cuda()
    lbls_sorted, _ = torch.sort(labels)
    lbls_alex_sorted, _ = torch.sort(labels_alex)
    assert torch.equal(lbls_sorted, lbls_alex_sorted)
    return image_batch_alex, image_batch, labels, orig_paths


image_batch_alex, image_batch, labels, orig_paths = None, None, None, None
attempt = 0
correct_imgs = -1
# find image that is classified correctly for all models and iters
while attempt < 50:
    print("attempt = {}".format(attempt))
    image_batch_alex_attempt, image_batch_attempt, \
            labels_attempt, orig_paths_attempt = get_image_both_sizes()
    attempt += 1
    failed = False
    for mod in models:

        # if one model failed, dont try others
        if failed is True:
            break

        print("*** model {} ***".format(mod))
        for it in iters:
            print("it {}".format(it))

            model = uf.get_model_arg(mod, it, longtrain=train_len == "long")
            model.eval()
            model.cuda()
            batch = image_batch_alex_attempt if "alex" in mod \
                                             else image_batch_attempt
            assert batch.shape[0] == BATCH_SIZE
            out = model(batch.cuda())
            predcat = torch.argmax(out, dim=1)
            correct_imgs = torch.flatten(
                torch.nonzero(predcat == labels_attempt))
            batch = batch[correct_imgs]
            labels = labels_attempt[correct_imgs]

            if "alex" in mod:
                image_batch_alex = batch
            else:
                image_batch = batch

            print("Remained {} imgs, need at least {}".format(
                len(correct_imgs), MIN_EXAMPLES))

            if len(correct_imgs) < MIN_EXAMPLES:
                print("failed!")
                failed = True
                break

    # if all models passed keep the batch
    if failed is False:
        break

if not len(correct_imgs) == image_batch.shape[0]:
    print(
        "len(correct_imgs)={} == image_batch.shape[0]={} == "
        "image_batch_alex.shape[0]={}".format(len(correct_imgs),
        image_batch.shape[0], image_batch_alex.shape[0]))
print("Batched picked!")

if attempt >= 50:
    print("Didn't find an image with correct classification!")
    exit()

rows = 1
cols = 2
figsize = (cols * 8, rows * 5)
fig, axs = plt.subplots(rows, cols, figsize=figsize)
mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
plt.subplots_adjust(left=None, bottom=None, right=None,
                    # top=1,
                    wspace=None, hspace=None)
plt.suptitle("# imgs {}".format(len(correct_imgs)), x=mid, y=1)

print("Starting attacks...")

for i, mod in enumerate(models):
    accs = []
    ax = axs[i]
    for it in iters:
        epsilons, robust_accuracy, raw, clipped, \
        is_adv, (predcat, adv_predcat, label) = \
            test.run_attacks(mod, it,
                             image_batch if "alexnet" not in mod
                                         else image_batch_alex,
                             labels,
                             attack_name="PGD",
                             longtrain=train_len == "long",
                             epsilons=epsilons)
        accs.append(robust_accuracy)
    ax.set_title(mod)
    for j, ep in enumerate(epsilons):
        ax.plot(iters, [x[j] for x in accs], label="ep {}".format(ep))

plt.ylim(top=1.0)
plt.legend()
plt.show()
plt.close()