def get_class_to_idx(path_dic=None):
    if path_dic is None:
        path_dic = "datasets/imagenet/1000_lbls.txt"
    class_to_idx = {}
    with open(path_dic) as f:
        for i, line in enumerate(f):
            sp = line.split(" ")
            cls = sp[0]
            index = sp[1]
            class_to_idx[cls] = index
    return class_to_idx

def get_img_to_class(path_dic=None):
    if path_dic is None:
        path_dic = "datasets/imagenet/val/val_annotations.txt"
    img_to_class = {}
    with open(path_dic) as f:
        for i, line in enumerate(f):
            out = line.split("	")[:2]
            img_to_class[out[0]] = out[1]
    return img_to_class

'''
cross domain transferrabe perturbations expects file of the form
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
...
'''
def create_standard_val_file_for_cross_dom_trans_perturb(path_dic=None):
    if path_dic is None:
        path_dic = "datasets/imagenet/val/val.txt"

    class_to_idx = get_class_to_idx()
    img_to_class = get_img_to_class()

    with open(path_dic, "w") as f:
        for img in img_to_class:
            idx = class_to_idx[img_to_class[img]]
            f.write("{} {}\n".format(img, idx))

    print("Created {}".format(path_dic))

def get_img_to_idx(class_to_idx_path=None, img_to_class_path=None):
    class_to_idx = get_class_to_idx(class_to_idx_path)
    img_to_class = get_img_to_class(img_to_class_path)
    img_to_idx = {}
    for img in img_to_class:
        img_to_idx[img] = class_to_idx[img_to_class[img]]
    return img_to_idx

def rename_imgs(path):
    import os
    imgs = os.listdir(path)
    imgs = [x for x in imgs if "jpeg" in x.lower()]
    print("images total ", len(imgs))
    for im in imgs:
        src = os.path.join(path, im)
        s = im.split(".")
        s[-1] = s[-1].lower()
        s = ".".join(s)
        dst = os.path.join(path, s)
        os.rename(src, dst)
        print("renamed {} to {}".format(src, dst))

#rename_imgs("/mnt/Vol2TBSabrentRoc/Projects/adversarial_research/datasets/ILSVRC2012_val_imagenet")

