
from torchvision.models import googlenet

from src import *

wandb.init(name='uniform_noise', project='attack_benchmark')

def clip(adv, img, eps):
    return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)

def attack(clean_image, model, eps=10 / 255, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    noise = ((2 * eps) * torch.rand(*clean_image.shape, device=device) - eps)
    noise.requires_grad = True

    out = clean_image
    return out


benchmark_attack(attack, googlenet(pretrained=True))
