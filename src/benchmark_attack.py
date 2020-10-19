import multiprocessing

import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
import torch
from src import ParallelImageFolders, CumulativeWandbLogger, DATASET_PATHS, get_normalization_transforms, get_loader
import torch.utils.data
from tqdm.auto import tqdm
import wandb


def benchmark_attack(attack, model, batch_size=64, subset=None, dataset='imagenet',
                     device=torch.device('cuda:0'), eps_sweep=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5), wandb_init=True):
    model.eval().to(device)
    logger = CumulativeWandbLogger()  # assume wandb init in main loop
    loader = get_loader(f'{DATASET_PATHS["imagenet"]}/val', batch_size=batch_size)
    forward_t, reverse_t = get_normalization_transforms(device=device)
    if wandb_init:
        wandb.init(name=attack.__name__, project='attack_benchmark', reinit=True)
    for eps in eps_sweep:
        count = 0
        for x, y, _ in tqdm(loader, desc='Validating', total=subset if subset is not None else len(loader)):
            x = x.to(device)
            y = y.to(device)
            clean_out = model(forward_t(x))
            _, clean_pred = torch.max(F.softmax(clean_out, dim=1), dim=1)
            adv_out = model(forward_t(attack(x, model, y, clean_out, forward_t, eps, device=device)))
            _, adv_pred = torch.max(F.softmax(adv_out, dim=1), dim=1)
            logger.add_metrics({'clean_accuracy': accuracy(clean_out, y).mean().detach().item(),
                                'adv_accuracy': accuracy(adv_out, y).mean().detach().item(),
                                'kl_divergence': F.kl_div(F.log_softmax(adv_out, dim=1),
                                                          F.softmax(clean_out, dim=1),
                                                          reduction='batchmean').detach().item()
                                })
            count += 1
            if subset is not None and count > subset:
                break
        logger.aggregate_and_log({'eps': eps})
