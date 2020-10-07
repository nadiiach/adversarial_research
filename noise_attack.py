from torchvision.models import googlenet

from src import *

wandb.init(name='uniform_noise', project='attack_benchmark')


def attack(clean_image, model, eps=10 / 255, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    noise = ((2 * eps) * torch.rand(*clean_image.shape, device=device) - eps)
    return clean_image + noise


benchmark_attack(attack, googlenet(pretrained=True))
