# Training program for alexnet.

import torch, os, argparse
# try:
#     from net.models import oldalexnet
#     from net.models import train_utils
# except:
#     import oldalexnet
#     import train_utils

from net.models import oldalexnet
from net.models import train_utils

torch.backends.cudnn.benchmark = True

# [int(0.5 + math.pow(2, 0.25 * i)) for i in range(100)]
snapshot_iters = [
   0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 19, 23, 27, 32, 38, 45,
   54, 64, 76, 91, 108, 128, 152, 181, 215, 256, 304, 362, 431, 512,
   609, 724, 861, 1024, 1218, 1448, 1722, 2048, 2435, 2896, 3444,
   4096, 4871, 5793, 6889, 8192, 9742, 11585, 13777, 16384, 19484,
   23170, 27554, 32768, 38968, 46341, 55109, 65536, 77936, 92682,
   110218, 131072, 155872, 185364, 220436, 262144, 311744, 370728,
   440872, 524288, 623487, 741455, 881744, 1048576, 1246974, 1482910,
   1763488, 2097152, 2493948, 2965821, 3526975, 4194304, 4987896,
]

def main(args):
    results_dir = 'results/{}{}/{}--init{}-batch{}-init_method{}' \
                  '-drop{}-groups{}-wd{}-optim{}-mom{}-iters{}-w_{}'.format(
        args.prefix, args.model, args.dataset, args.init_lr, args.batch_size,
        args.init_method,
        args.dropout, args.groups,
        args.weight_decay, args.optimizer, args.momentum,
        args.iters, str(args.w) if len(args.w) > 0 else "standard"
    )
    
    print("results_dir={}".format(results_dir))

    training_dir = '%s/%s/train' % (args.ds_root, args.dataset)
    val_dir = '%s/%s/val' % (args.ds_root, args.dataset)
    num_classes=dict(places=365, imagenet=1000)[args.dataset]
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'args.txt'), 'w') as f:
        f.write(str(args) + '\n')

    train_utils.printstat(str(args), results_dir)

    train_loader, val_loader = train_utils.get_loders(256, 227, 48, 24,
                                                      training_dir, val_dir,
                                                      args)

    print("DataLoader created")
    # We initialize the model with some random, some identity convolutions.
    # TODO: consider setting include_lrn=False here. (Left it on by accident.)
    model = oldalexnet.AlexNet(num_classes=num_classes,
                               include_dropout=args.dropout,
                               split_groups=args.groups,
                               w=args.w)

    apply_init(model)
    model.train()
    model.cuda()

    max_iter = args.iters + 1

    criterion, optimizer, scheduler = train_utils.get_crit_opt_sched(model,
                                                                     max_iter,
                                                                     args)

    iter_num = 0
    best = dict(val_accuracy=0.0)
    checkpoint_filename = 'weights.pth'
    best_filename = 'best_%s' % checkpoint_filename
    best_checkpoint = os.path.join(results_dir, best_filename)
    try_to_resume_training = args.resume is not None

    if try_to_resume_training and os.path.exists(best_checkpoint):
        iter_num = train_utils.resume_training(args, best_filename, results_dir,
                        model, optimizer, scheduler, best)

    # Track the average training loss/accuracy between snapshots.
    train_loss = train_utils.AverageMeter()
    train_acc = train_utils.AverageMeter()

    # Here is our training loop.
    while iter_num < max_iter:
        for t_input, t_target in train_loader:
            input_var, target_var = [d.cuda() for d in [t_input, t_target]]
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss.update(loss.data.item(), t_input.size(0))

            if iter_num > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    t_input.size(0))
            train_acc.update(accuracy)
            remaining = 1 - iter_num / float(max_iter)

            keep = args.keep and (iter_num in snapshot_iters
                    or iter_num > 200 and (iter_num + 100 in snapshot_iters)
                    or iter_num == max_iter - 1)
            if keep or iter_num % 1000 == 0:
                train_utils.validate_and_checkpoint(keep, model, val_loader,
                                                    criterion, iter_num,
                                                    optimizer, scheduler,
                                                    train_acc, train_loss, best,
                                                    results_dir,
                                                    checkpoint_filename,
                                                    best_filename
                                                    )
                model.train()
                if iter_num % 1000 == 0:
                    train_loss.reset()
                    train_acc.reset()

            iter_num += 1
            if iter_num >= max_iter:
                break


def apply_init(model):
    for n, p in model.named_parameters():
        if 'bias' in n:
            torch.nn.init.zeros_(p)
        else:
            torch.nn.init.kaiming_normal_(p, nonlinearity='relu')


if __name__ == '__main__':
    def parseargs():
        parser = argparse.ArgumentParser()

        def aa(*args, **kwargs):
            parser.add_argument(*args, **kwargs)

        aa('--dataset', choices=['places', 'imagenet'], default='places')
        aa('--iters', type=int, default=snapshot_iters[-1])
        aa('--init_lr', type=float, default=2e-2)
        aa('--weight_decay', type=float, default=5e-4)
        aa('--ds_root', default='datasets')
        aa('--batch_size', type=int, default=256)
        aa('--init_method', choices=['k'], default='k')
        aa('--optimizer', choices=['adam', 'sgd'], default='sgd')
        aa('--momentum', type=float, default=0.9)
        aa('--dropout', type=int, choices=[0, 1], default=1)
        aa('--groups', type=int, choices=[0, 1], default=1)
        aa('--keep', type=int, default=1)
        aa('--resume', type=str, default="best")
        aa('--prefix', type=str, default='')
        aa('--w', nargs="+", type=int, default=[])
        args = parser.parse_args()
        return args

    args = parseargs()
    main(args)