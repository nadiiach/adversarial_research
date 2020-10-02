import foolbox as fb
from utils import util_functions as uf
import foolbox.criteria as fc
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join
from utils import logger
import numpy as np
from utils import util_functions as uf
import torch
import cv2

logger = logger.get_logger()
BATCH_SIZE = 1

def get_batch(model, dataset="places", batch_size=1):
    crop_size = 227 if "alexnet" in model else 224
    base = uf.find_child_dir_path("datasets")
    path = join(base, dataset, "val")
    dl_train = uf.get_dataloader(path, crop_size, batch_size=batch_size,
                                 num_workers=0, logger=logger,
                                 normalized_version=False)

    image_batch, labels, orig_paths, \
        names, cats, orig_images = uf.get_batch_of_images(dl_train)
    image_batch = image_batch.cuda()
    labels = labels.cuda()
    catlist = dl_train.dataset.classes
    return image_batch, labels, orig_paths, names, cats, orig_images, catlist


def run_attacks(model, it, image_batch, labels,
                attack_name="FGSM", epsilons=None, longtrain=True):
    success = True
    attacks = {
        "FGSM": fa.FGSM(),
        "PGD": fa.LinfPGD(),
        "BasicIterativeAttack": fa.LinfBasicIterativeAttack(),
        "AdditiveUniformNoiseAttack": fa.LinfAdditiveUniformNoiseAttack(),
        "DeepFoolAttack": fa.LinfDeepFoolAttack(),
    }
    bounds = (-2.2, 2.7)

    model = uf.get_model_arg(model, it, logger=logger, longtrain=longtrain)
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=bounds)
    fmodel = fmodel.transform_bounds(bounds)
    assert fmodel.bounds == bounds
    acc = fb.utils.accuracy(fmodel, image_batch, labels)

    logger.debug("Default accuracy {}".format(acc))

    if epsilons is None:
        epsilons = [0.001]

    logger.debug("\nStarting {}".format(attack_name))
    attack_object = attacks[attack_name]
    logger.debug("Running attack with epsilons={}".format(epsilons))

    raw, clipped, is_adv = attack_object(fmodel, image_batch,
                                         labels, epsilons=epsilons)

    robust_accuracy = 1 - is_adv.cpu().numpy().mean(axis=-1)
    # for ep, ac in zip(epsilons, robust_accuracy):
    #     print("epsilon {} accuracy {}".format(ep, ac))

    # sanity check
    model.cuda()
    out = model(image_batch.cuda())
    predcat = torch.argmax(out).cpu().numpy()

    out = model(clipped[0].cuda())
    adv_predcat = torch.argmax(out).cpu().numpy()
    label = labels.cpu().numpy()[0]

    return epsilons, robust_accuracy, raw, clipped, \
            is_adv, (predcat, adv_predcat, label)



def run_long_model():
    # dic = uf.get_available_iters_and_weights_paths("resnet18-forever")
    # print(dic)
    # print(sorted(dic.keys()))
    # exit()
    #attacks = ["FGSM", "PGD", "AdditiveUniformNoiseAttack", "DeepFoolAttack"]
    attacks = ["FGSM", "PGD", "BasicIterativeAttack", "DeepFoolAttack"]
    global logger
    global BATCH_SIZE
    model = "resnet18-forever"
    image_batch, labels, orig_paths, names, cats, \
        orig_images, catlist = get_batch(model, batch_size=BATCH_SIZE)
    if model == "resnet18-long":
        iters = [32668, 65436, 131072, 185264, 262044,
                 370628, 524188, 881644, 1048577]
    if model == "resnet18-forever":
        iters = [65436, 131072, 185264, 262044,
                 370628, 524188, 881644, 1048576, 1763488]

    # iters = [32668]
    rows = 2
    cols = 2
    figsize = (cols * 8, rows * 5)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                        wspace=None, hspace=None)
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle("Attacks on {}".format(model), x=mid, y=1)

    i = 0
    for r in range(rows):
        for c in range(cols):
            print(len(attacks), i)
            attack = attacks[i]
            i += 1
            ax = axs[r][c]
            ax.set_ylim(0, 0.6)
            ax.set_title('Attack {}'.format(attack))
            ax.set_xlabel('Epsilons')
            ax.set_ylabel('Accuracy')


            for it in iters:
                print("Processing row={} col={}".format(r, c))
                epsilons, robust_accuracy, raw, clipped, \
                    is_adv, (predcat, adv_predcat, label) = \
                    run_attacks(model, it, image_batch, labels,
                                attack_name=attack)
                logger.debug("epsilons: {}".format(epsilons))
                logger.debug("robust_accuracy: {}".format(robust_accuracy))

                ax.plot(epsilons, robust_accuracy, label=str(it))

            ax.legend()

    plt.show()


def run_compare_four_models_func_epsilon(mods, longtrain=True):
    # dic = uf.get_available_iters_and_weights_paths("resnet18-forever")
    # print(dic)
    # print(sorted(dic.keys()))
    # exit()
    attacks = ["FGSM", "PGD", "BasicIterativeAttack", "DeepFoolAttack"]
    global logger

    assert len(mods) == 4
    image_batch, labels, orig_paths, \
        names, cats, orig_images, catlist = get_batch("vgg16",
                                                      batch_size=BATCH_SIZE)
    image_batch_alex, labels_alex, orig_paths_alex, \
        names_alex, cats_alex, orig_images_alex, catlist = get_batch("alexnet",
                                                      batch_size=BATCH_SIZE)

    it = 131072

    #iters = [32668]
    rows = 2
    cols = 2
    figsize = (cols * 8, rows * 5)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                        wspace=None, hspace=None)
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle("Adversarial attacks", x=mid, y=1)

    i = 0
    for r in range(rows):
        for c in range(cols):
            print(len(attacks), i)
            attack = attacks[i]
            i += 1
            ax = axs[r][c]
            ax.set_ylim(0, 0.7)
            ax.set_title('Attack {}'.format(attack))
            ax.set_xlabel('Epsilons')
            ax.set_ylabel('Accuracy')

            for model in mods:
                if "alexnet" in model:
                    imb = image_batch_alex
                    lbls = labels_alex
                else:
                    imb = image_batch
                    lbls = labels

                print("Processing row={} col={}".format(r, c))

                epsilons, robust_accuracy, \
                    raw, clipped, is_adv, _ = run_attacks(model, it,
                                                          imb, lbls,
                                                          attack_name=attack,
                                                          longtrain=longtrain)
                print("epsilons ", epsilons)
                print("robust_accuracy ", robust_accuracy)

                logger.debug("epsilons: {}".format(epsilons))
                logger.debug("robust_accuracy: {}".format(robust_accuracy))
                ax.plot(epsilons, robust_accuracy, label=str(model))

            ax.legend()
    plt.show()

def run_compare_four_models_func_iters(mods, longtrain=True):
    # dic = uf.get_available_iters_and_weights_paths("resnet18-forever")
    # print(dic)
    # print(sorted(dic.keys()))
    # exit()
    attacks = ["FGSM", "PGD", "BasicIterativeAttack", "DeepFoolAttack"]
    global logger

    assert len(mods) == 4
    image_batch, labels, orig_paths, \
        names, cats, orig_images, catlist = get_batch("vgg16",
                                                      batch_size=BATCH_SIZE)
    image_batch_alex, labels_alex, orig_paths_alex, \
        names_alex, cats_alex, orig_images_alex, catlist = get_batch("alexnet",
                                                      batch_size=BATCH_SIZE)

    it = 131072

    #iters = [32668]
    rows = 2
    cols = 2
    figsize = (cols * 8, rows * 5)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                        wspace=None, hspace=None)
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle("Adversarial attacks", x=mid, y=1)

    i = 0
    for r in range(rows):
        for c in range(cols):
            print(len(attacks), i)
            attack = attacks[i]
            i += 1
            ax = axs[r][c]
            ax.set_ylim(0, 0.7)
            ax.set_title('Attack {}'.format(attack))
            ax.set_xlabel('Epsilons')
            ax.set_ylabel('Accuracy')

            for model in mods:
                if "alexnet" in model:
                    imb = image_batch_alex
                    lbls = labels_alex
                else:
                    imb = image_batch
                    lbls = labels

                print("Processing row={} col={}".format(r, c))

                epsilons, robust_accuracy, \
                    raw, clipped, is_adv, _ = run_attacks(model, it,
                                                          imb, lbls,
                                                          attack_name=attack,
                                                          longtrain=longtrain)
                print("epsilons ", epsilons)
                print("robust_accuracy ", robust_accuracy)

                logger.debug("epsilons: {}".format(epsilons))
                logger.debug("robust_accuracy: {}".format(robust_accuracy))
                ax.plot(epsilons, robust_accuracy, label=str(model))

            ax.legend()
    plt.show()

def run_func_of_iters():
    attacks = ["FGSM", "PGD", "BasicIterativeAttack", "DeepFoolAttack"]
    global logger

    model = "resnet18-forever"
    image_batch, labels, orig_paths, \
        names, cats, orig_images, catlist = get_batch(model,
                                                      batch_size=BATCH_SIZE)
    if model == "resnet18-long":
        iters = [32668, 65436, 131072, 185264, 262044, 370628,
                 524188, 881644, 1048577]
    if model == "resnet18-forever":
        iters = [65436, 131072, 185264, 262044, 370628, 524188,
                 881644, 1048576, 1763488]

    #iters = [32668]
    rows = 2
    cols = 2
    figsize = (cols * 8, rows * 5)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                        wspace=None, hspace=None)
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle("Attacks on {}".format(model), x=mid, y=1)

    i = 0
    for r in range(rows):
        for c in range(cols):
            print(len(attacks), i)
            attack = attacks[i]
            i += 1
            ax = axs[r][c]
            ax.set_ylim(0, 0.6)

            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')

            robaccuracies = []
            eps = -1

            for it in iters:
                epsilons, robust_accuracy, raw, clipped, is_adv, _ = \
                    run_attacks(model, it, image_batch, labels,
                                attack_name=attack, epsilons=[0.001])
                robaccuracies.append(robust_accuracy[0])
                eps = epsilons[0]

            ax.plot(iters, robaccuracies)
            ax.set_title('Model {}, Attack {}, Epsilon {}'.format(model,
                                                                  attack, eps))

    plt.show()


def run_func_of_iters_many_eps():
    # attacks = ["FGSM", "PGD", "BasicIterativeAttack", "DeepFoolAttack"]
    attacks = ["FGSM"]
    global logger

    model = "resnet18-long"
    image_batch, labels, orig_paths, \
        names, cats, orig_images, catlist = get_batch(model,
                                                      batch_size=BATCH_SIZE)
    if model == "resnet18-long":
        iters = [8092, 16384, 32668, 65436, 131072, 185264, 262044, 370628,
                 524188, 881644, 1048577]
    if model == "resnet18-forever":
        iters = [8092, 16384, 32668, 65436, 131072, 185264, 262044, 370628,
                 524188, 881644, 1048576, 1763488]

    import math
    #iters = [32668]
    rows = int(math.sqrt(len(attacks)))
    cols = int(math.sqrt(len(attacks)))

    figsize = (cols * 8, rows * 5)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                        wspace=None, hspace=None)
    #mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.title("Model {}, Attacks {}".format(model, attacks))

    epsilons = [0, 0.00003, 0.00008, 0.0001, 0.0003, 0.0008,
                0.001, 0.003, 0.008, 0.01, 0.03, 0.08, 0.1]
    #
    # epsilons = [0]

    i = 0
    for r in range(rows):
        for c in range(cols):

            if rows > 1:
                ax = axs[r][c]
            elif cols > 1:
                ax = axs[c]
            else:
                ax = axs

            ax.set_ylim(0, 0.6)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')

            for eps in epsilons:
                attack = attacks[i]
                robaccuracies = []
                for it in iters:
                    epsilons, robust_accuracy, raw, clipped, is_adv, _ = \
                        run_attacks(model, it, image_batch, labels,
                                    attack_name=attack, epsilons=[eps])
                    robaccuracies.append(robust_accuracy[0])
                    eps = epsilons[0]

                ax.set_xscale('log')
                ax.plot(iters, robaccuracies, label="eps:{}".format(str(eps)))
            ax.legend()
        i += 1

    plt.legend(bbox_to_anchor=(0, 0, -0.1, 0.8), loc='upper right',
               borderaxespad=0.)
    # fig.subplots_adjust(top=0.5)
    plt.show()


def single_image_attach(input=None, epsilon=0.01, attack_name="PGD", it=131072,
                        model_to_attack="vgg16", plot=False, single_try=False,
                        catlist=None):
    #print("Running model attack...")
    model = model_to_attack


    if input is None:
        image_batch, labels, orig_paths, \
            names, cats, orig_images, catlist = get_batch(model,
                                                          batch_size=BATCH_SIZE)
        logger.debug("Input is None, sampled new input {}".format(names[0]))
    else:
        image_batch, labels, orig_paths, names, cats, orig_images = input
        assert len(input) == 6

    i = 0

    prev_img_path = orig_paths[0]

    while 1:
        i += 1
        epsilons, robust_accuracy, raw, clipped, is_adv, \
            (predcat, adv_predcat, label) = run_attacks(model, it,
                                                   image_batch, labels,
                                                   attack_name=attack_name,
                                                   epsilons=[epsilon])
        prev_img_path = orig_paths[0]

        if single_try:
            if predcat != label:
                raise Exception("Network made mistake on original input!")
            break

        if predcat != label:
            print("Attempt ", i)
            logger.error("Network misclassified original...")

            while 1:
                print("Getting new image...")
                image_batch, labels, orig_paths, \
                    names, cats, orig_images, catlist = get_batch(model,
                                                        batch_size=BATCH_SIZE)
                if orig_paths[0] != prev_img_path:
                    break


    # print("After attack accuracy ", robust_accuracy[0])
    # print("Torch...")
    # print("Img+clipped noise min={} max={}".format(torch.min(clipped[0]),
    #                                        torch.max(clipped[0])))
    #
    # print("image_batch min={} max={}".format(torch.min(image_batch),
    #                                        torch.max(image_batch)))

    noise_clipped = image_batch - clipped[0]
    img = uf.get_image_from_tensor(image_batch)
    img_wnoise = uf.get_image_from_tensor(clipped[0])
    noise_clipped = uf.get_image_from_tensor(noise_clipped)

    # print("noise_clipped min={} max={}".format(np.min(noise_clipped),
    #                                            np.max(noise_clipped)))

    normalizedImg = np.zeros(img.shape)
    img = cv2.normalize(img, normalizedImg,
                                 0, 255, cv2.NORM_MINMAX)

    normalizedImgWNoise = np.zeros(img_wnoise.shape)
    img_wnoise = cv2.normalize(img_wnoise, normalizedImgWNoise,
                                 0, 255, cv2.NORM_MINMAX)

    normalizedNoise = np.zeros(noise_clipped.shape)
    noise_clipped = cv2.normalize(noise_clipped, normalizedNoise,
                                 0, 255, cv2.NORM_MINMAX)

    img = img.astype(np.uint8)
    img_wnoise = img_wnoise.astype(np.uint8)
    noise = noise_clipped.astype(np.uint8)

    # print("-----")
    # print("img_ min={} max={}".format(np.min(img), np.max(img)))
    #
    # print("img_wnoise_ min={} max={}".format(np.min(img_wnoise),
    #                                          np.max(img_wnoise)))
    #
    # print("noise_ min={} max={}".format(np.min(noise), np.max(noise)))

    suptitle = "Dataset: Places, Name:{}, " \
               "Class:{}, Model:{}, Attack:{}, Eps:{}, " \
               "Correct orig?:{}, Attack success?:{}, Advcat: {}".format(
                    names[0], cats[0], model, attack_name, epsilons[0],
                    predcat == label, adv_predcat != label,
                    catlist[adv_predcat])
    if plot:
        cols = 3
        rows = 1
        figsize = (cols * 7, rows * 5)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        axs[0].imshow(img)
        axs[0].axis("off")
        axs[1].imshow(img_wnoise)
        axs[1].axis("off")
        axs[2].imshow(noise)
        axs[2].axis("off")

        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        plt.subplots_adjust(left=None, bottom=None, right=None, top=1.1,
                            wspace=None, hspace=None)
        mpl.rcParams["figure.titlesize"] = 'large'
        plt.suptitle(suptitle, x=mid, y=1)
        plt.show()

    # clipped is orig image + clipped noise
    assert clipped[0].shape == image_batch.shape
    return clipped[0], (img, img_wnoise, noise), \
            suptitle, adv_predcat != label, catlist, adv_predcat
