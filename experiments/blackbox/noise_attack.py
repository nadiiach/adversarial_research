import sys

sys.path.append("../..")

from torchvision.models import googlenet

from src import *
import argparse

parser = argparse.ArgumentParser(prog="Benchmark Noise Attack", description="")
parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int)
args = parser.parse_args()


def uniform_noise_attack(clean_image, model, y, forward_t, eps, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    noise = ((2 * eps) * torch.rand(*clean_image.shape, device=device) - eps)
    return clean_image + noise


benchmark_attack(uniform_noise_attack, googlenet(pretrained=True),
                 device=torch.device(f'cuda:{args.gpu}'),
                 subset=5)  # only test on 20 batches of data
