from adversarial_attacks.test import run_long_model, run_func_of_iters, \
    run_func_of_iters_many_eps, single_image_attach, get_batch
from utils import util_functions as uf
import matplotlib.pyplot as plt
import torch

ITERATION = 131072
TRANS_MOD = "resnet50"
WB_MOD = "resnet18"
ATTACKTYPE = "PGD"
max_attempts = 15  # if networks makes original miscalssification
images_to_test = 1000
# run_func_of_iters_many_eps()
# run_long_model()
# run_compare_models()
epsils = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
i = 0

transfer_model = uf.get_model_arg(TRANS_MOD, ITERATION)
transfer_model.cuda()
transfer_model.eval()

f = open("attack_and_transfer_{}_to_{}.csv".format(WB_MOD, TRANS_MOD), "w")
f.write("img,lbl,eps,pred_attack_wb_model,pred_tf_model,"
        "pred_attack_tf_model,transfered?,wb_model,tfmodel,it\n")


for trial in range(images_to_test):
    while i < max_attempts:
        #print("Attempt {}".format(i))
        try:
            image_batch, labels, orig_paths, \
                names, cats, orig_images, catlist = get_batch(WB_MOD)
            inp = (image_batch, labels, orig_paths, names, cats, orig_images)

            for i, eps in enumerate(epsils):
                clipped, (img, img_wnoise, noise), \
                    suptitle, attack_success, catlist, adv_predcat = \
                    single_image_attach(input=inp, epsilon=eps,
                                        attack_name=ATTACKTYPE,
                                        it=ITERATION, model_to_attack=WB_MOD,
                                        plot=False, single_try=True,
                                        catlist=catlist)

                # tf model test
                out = transfer_model(image_batch.cuda())
                tf_pred = torch.argmax(out).cpu().numpy()

                # transfer attack
                out = transfer_model(clipped)
                tf_adv_pred = torch.argmax(out).cpu().numpy()
                label = labels.cpu().numpy()[0]

                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                    names[0], catlist[label], eps, catlist[adv_predcat],
                    catlist[tf_pred], catlist[tf_adv_pred],
                    (tf_pred == label and label != tf_adv_pred), WB_MOD,
                    TRANS_MOD, ITERATION
                ))
            f.flush()
            break
        except Exception as e:
            i += 1
            print("ERROR: ", e)
            #print("Trying new image...")

f.close()