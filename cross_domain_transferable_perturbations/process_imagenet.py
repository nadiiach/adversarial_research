
def get_class_to_idx(path_dic=None):
    if path_dic is None:
        path_dic = "datasets/imagenet/1000_lbls.txt"
    class_to_idx = {}
    with open(path_dic) as f:
        for i, line in enumerate(f):
            cls = line.split(" ")[0][:-1]
            class_to_idx[cls] = i
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


