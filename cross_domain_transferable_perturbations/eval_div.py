import argparse
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
try:
    from cross_domain_transferable_perturbations.gaussian_smoothing import *
    from cross_domain_transferable_perturbations.utils import *
    from cross_domain_transferable_perturbations.process_imagenet import *
except:
    from gaussian_smoothing import *
    from utils import *
    from process_imagenet import *

def main():
    parser = argparse.ArgumentParser(description='Cross Data Transferability')
    parser.add_argument('--test_dir', default='../datasets/imagenet/val',
                        help='ImageNet Validation Data')
    parser.add_argument('--is_nips', action='store_true', help='Evaluation on NIPS data')
    parser.add_argument('--measure_adv', action='store_true',
                        help='If not set then measuring only clean accuracy', default=True)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=9, help='Which Saving Instance to Evaluate')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
    parser.add_argument('--model_type', type=str, default='vgg16',
                        help='Model against GAN is trained: vgg16, '
                             'vgg19, incv3, res152')
    parser.add_argument('--model_t', type=str, default='res152',
                        help='Model under attack : vgg16, vgg19, '
                             'incv3, res152, res50, dense201, sqz')
    parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
    parser.add_argument('--attack_type', type=str, default='img',
                        help='Training is either img/noise dependent')
    parser.add_argument('--gk', action='store_true',
                        help='Apply Gaussian Smoothings to GAN Output')
    parser.add_argument('--rl', action='store_true',
                        help='Relativstic or Simple GAN', default=True)
    parser.add_argument('--attempts', type=int,
                        help='How many times try to attack', default=10)
    parser.add_argument('--foldname', type=str, required=True,
                        help="From what folder should you load GAN?")
    args = parser.parse_args()
    print(args)

    # Normalize (0-1)
    eps = args.eps / 255
    print("eps {}".format(eps))

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Input dimensions: Inception takes 3x299x299
    if args.model_type in ['vgg16', 'vgg19', 'res152']:
        scale_size = 256
        img_size = 224
    else:
        scale_size = 300
        img_size = 300 #299

    netG = load_gan(args, args.foldname, channels=4)
    netG.to(device)
    netG.eval()

    model_t = load_model(args)
    model_t = model_t.to(device)
    model_t.eval()

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return t

    test_dir = args.test_dir
    #test_dir ="/mnt/Vol2TBSabrentRoc/Projects/adversarial_research/datasets/imagenet/val"
    print(os.path.dirname(os.path.realpath(__file__)))
    test_set = datasets.ImageFolder(test_dir, data_transform)
    test_size = len(test_set)
    print('Test data size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    # Setup-Gaussian Kernel
    if args.gk:
        kernel_size = 3
        pad = 2
        sigma = 1
        kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad,
                                     sigma=sigma).cuda()

    # Evaluation
    adv_acc = 0
    clean_acc = 0
    disagreement_rate = 0

    print("Test loader {}".format(test_loader))

    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)

        imgs_repeat = args.attempts
        img = torch.cat([img] * imgs_repeat)
        label = torch.cat([label] * imgs_repeat)

        s = img.shape
        noise = torch.rand((s[0], 1, s[2], s[3]))
        assert not torch.all(noise[0].eq(noise[1]))
        img = img.to(device)
        noise = noise.to(device)
        imgwnoise = torch.cat((img, noise), 1)
        imgwnoise = imgwnoise.to(device)
        clean_out = model_t(normalize(img.clone().detach()))
        out = torch.sum(clean_out.argmax(dim=-1) == label).item()
        clean_res = 1 if out >= 1 else 0
        clean_acc += clean_res
        # Unrestricted Adversary
        adv = netG(imgwnoise)

        # Apply smoothing
        if args.gk:
            adv = kernel(adv)

        # Projection
        adv_noise = adv.clone()

        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        adv_out = model_t(normalize(adv.clone().detach()))
        adv_sum = torch.sum(adv_out.argmax(dim=-1) == label).item()
        adv_res = 0 if adv_sum < args.attempts else 1
        adv_acc += adv_res
        fool_out = torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()
        disagreement_rate += 1 if fool_out >= 1 else 0

        if i % 1000 == 0:
            rooty = "saved_models/{}/images".format(args.foldname)
            if not os.path.exists(rooty):
                os.mkdir(rooty)

            vutils.save_image(
                vutils.make_grid(adv_noise, normalize=True, scale_each=True),
                                '{}/noise_{}.png'.format(rooty, i))
            vutils.save_image(
                vutils.make_grid(adv, normalize=True, scale_each=True),
                                '{}/adv_{}.png'.format(rooty, i))
            vutils.save_image(
                vutils.make_grid(img, normalize=True, scale_each=True),
                                '{}/org_{}.png'.format(rooty, i))

        if i % 100 == 0:
            if args.measure_adv:
                print('At Batch:{}\t l_inf:{}'
                      .format(i, (img - adv).max() * 255))
            else:
                print('At Batch:{}'.format(i))

        if i == 10000:
            test_size = 10000
            break

    cleanaccuracy = clean_acc / test_size
    aversarialaccuracy = adv_acc / test_size
    disagreement = disagreement_rate / test_size
    foolrate = 1 - (aversarialaccuracy/cleanaccuracy)

    print('Clean:{0:.3%}\t Adversarial :{1:.3%}\t Fool Rate :{2:.3%}\t Disagreement Rate:{3:.3%}'
        .format(cleanaccuracy, aversarialaccuracy, foolrate, disagreement))


if __name__ == "__main__":
    main()