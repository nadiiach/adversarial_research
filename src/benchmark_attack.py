import multiprocessing

import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
import torch
from src import ParallelImageFolders, CumulativeWandbLogger, DATASET_PATHS, get_normalization_transforms, get_loader
import torch.utils.data
from tqdm.auto import tqdm


def benchmark_attack(attack, model, batch_size=64, subset=False, dataset='imagenet', device=torch.device('cuda:0')):
    model.eval().to(device)
    logger = CumulativeWandbLogger()  # assume wandb init in main loop
    loader = get_loader(f'{DATASET_PATHS["imagenet"]}/val', batch_size=batch_size)
    forward_t, reverse_t = get_normalization_transforms(device=device)
    count = 0
    for x, y, _ in tqdm(loader, desc='Validating', total=20 if subset else len(loader)):
        x = x.to(device)
        y = y.to(device)
        clean_out = model(forward_t(x))
        _, clean_pred = torch.max(F.softmax(clean_out, dim=1), dim=1)
        adv_out = model(forward_t(attack(x, model, y, forward_t, device=device)))
        _, adv_pred = torch.max(F.softmax(adv_out, dim=1), dim=1)
        logger.add_metrics({'clean_accuracy': accuracy(clean_out, y).mean().detach().item(),
                            'adv_accuracy': accuracy(adv_out, y).mean().detach().item(),
                            'kl_divergence': F.kl_div(F.log_softmax(adv_out, dim=1),
                                                      F.softmax(clean_out, dim=1),
                                                      reduction='batchmean').detach().item()
                            })
        count += 1
        if subset and count > 20:
            break
    logger.aggregate_and_log()
