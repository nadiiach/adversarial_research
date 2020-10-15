import sys

sys.path.append("../..")

from torchvision.models import *
import argparse
from src import *

parser = argparse.ArgumentParser(prog="Benchmark PGD Attack", description="")
parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int)

parser.add_argument('-n', '--num_steps', default=40, dest='num_steps', type=int)
parser.add_argument('-s', '--step_size', default=0.01, dest='step_size', type=float)
parser.add_argument('-e', '--eps', default=5 / 255, dest='eps', type=float)

args = parser.parse_args()


def clip(adv, img, eps):
    return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)


wandb.init(name='whitebox_pgd', project='attack_benchmark')

wandb.log({'num_steps': args.num_steps,
           'step_size': args.step_size,
           'eps': args.eps})


class Ensemble():
    def __init__(self):

        self.ensemble = [vgg16(pretrained=True),
                         resnet152(pretrained=True),
                         resnet101(pretrained=True),
                         resnet50(pretrained=True),
                         mobilenet_v2(pretrained=True)]

        for net in self.ensemble:
            net.eval()

    def get_loss(self, x, y):
        loss = F.cross_entropy(self.ensemble[0](x), y)
        for net in self.ensemble[1:]:
            loss += F.cross_entropy(net(x), y)
        return loss


ensemble = Ensemble()


def attack(clean_image, model, y, forward_t, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    adv = clean_image + ((2 * args.eps) * torch.rand(*clean_image.shape, device=device) - args.eps)
    for i in range(args.num_steps):
        adv = clip(adv, clean_image, args.eps).detach()  # project onto valid space
        adv.requires_grad = True
        loss = ensemble.get_loss(forward_t(adv), y)
        loss.backward()
        adv = clip(adv + args.step_size * torch.sign(adv.grad.data), clean_image, args.eps).detach()

    return clip(adv, clean_image, args.eps)


benchmark_attack(attack, googlenet(pretrained=True), subset=False, device=torch.device('cuda:2'))
