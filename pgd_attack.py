from torch.optim import Adam
from torchvision.models import googlenet

from src import *

wandb.init(name='whitebox_pgd', project='attack_benchmark')


def clip(adv, img, eps):
    return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)


num_steps = 40
step_size = 0.01
wandb.log({'num_steps': num_steps,
           'step_size': step_size})


def attack(clean_image, model, y, forward_t, eps=10 / 255, device='cpu'):
    """
    Generate adversarial image from clean images (Bx3xWxH) and model (if whitebox attack).
    :param clean_image: original image
    :param model: model (neural network) to optimize
    :param eps: perturbation budget (in pretransformed space)
    :return:
    """
    adv = clean_image + ((2 * eps) * torch.rand(*clean_image.shape, device=device) - eps)
    for i in range(num_steps):
        adv = clip(adv, clean_image, eps).detach()  # project onto valid space
        adv.requires_grad = True
        loss = F.cross_entropy(model(forward_t(adv)), y)
        loss.backward()
        adv = clip(adv + step_size * torch.sign(adv.grad.data), clean_image, eps).detach()

    return clip(adv, clean_image, eps)


benchmark_attack(attack, googlenet(pretrained=True), subset=False, device=torch.device('cuda:2'))
