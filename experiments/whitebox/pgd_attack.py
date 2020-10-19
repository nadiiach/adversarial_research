import sys

sys.path.append("../..")

from src import *
import argparse
from torchvision.models import googlenet

parser = argparse.ArgumentParser(prog="Benchmark PGD Attack", description="")
parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int)
parser.add_argument('-n', '--num_steps', default=40, dest='num_steps', type=int)
parser.add_argument('-s', '--step_size', default=0.01, dest='step_size', type=float)

args = parser.parse_args()


def clip(adv, img, eps):
    return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)


wandb.init(name='whitebox_pgd_attack', project='attack_benchmark')

wandb.log({'num_steps': args.num_steps,
           'step_size': args.step_size})


def whitebox_pgd_attack(clean_image, model, y, forward_t, eps, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    adv = clean_image + ((2 * eps) * torch.rand(*clean_image.shape, device=device) - eps)
    for i in range(args.num_steps):
        adv = clip(adv, clean_image, eps).detach()  # project onto valid space
        adv.requires_grad = True
        loss = F.cross_entropy(model(forward_t(adv)), y)
        loss.backward()
        adv = clip(adv + args.step_size * torch.sign(adv.grad.data), clean_image, eps).detach()

    return clip(adv, clean_image, eps)


benchmark_attack(whitebox_pgd_attack, googlenet(pretrained=True), subset=5, device=torch.device(f'cuda:{args.gpu}'),
                 wandb_init=False)
