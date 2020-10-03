# Training program for alexnet.

import torch, numpy, os, shutil, math, re, torchvision, argparse
from netdissect import parallelfolder, renormalize, pbar
from torchvision import transforms
from net.models import train_utils
torch.backends.cudnn.benchmark = True
from net.models import softresnet

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
    if args.model == 'resnet50':
        print("Training resnet50...")
    else:
        print("Training resnet18...")

    results_dir = 'results/{}{}/{}--init{}-batch{}-init_method{}' \
                  '-drop{}-groups{}-wd{}-optim{}-mom{}-iters{}-w_{}'.format(
        args.prefix, args.model, args.dataset, args.init_lr, args.batch_size,
        args.init_method,
        args.dropout, args.groups,
        args.weight_decay, args.optimizer, args.momentum,
        args.iters, str(args.w) if len(args.w) > 0 else "standard"
    )

    training_dir = '%s/%s/train' % (args.ds_root, args.dataset)
    val_dir = '%s/%s/val' % (args.ds_root, args.dataset)
    num_classes = dict(places=365, imagenet=1000)[args.dataset]
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'args.txt'), 'w') as f:
        f.write(str(args) + '\n')

    train_utils.printstat(str(args), results_dir)

    train_loader, val_loader = train_utils.get_loders(256, 224, 48, 24,
                                                      training_dir, val_dir,
                                                      args)

    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.model == 'softresnet18':
        model = softresnet.resnet18(num_classes=num_classes)
    elif args.model == 'softresnet50':
        model = softresnet.resnet50(num_classes=num_classes)

    apply_init(model, args.init_method)
    model.train()
    model.cuda()

    max_iter = args.iters + 1

    criterion, optimizer, scheduler = train_utils.get_crit_opt_sched(model,
                                                                     max_iter,
                                                                     args)

    iter_num = 0
    best = dict(val_accuracy=0.0)
    # Oh, hold on.  Let's actually resume training if we already have a model.
    checkpoint_filename = 'weights.pth'
    best_filename = 'best_%s' % checkpoint_filename
    best_checkpoint = os.path.join(results_dir, best_filename)
    try_to_resume_training = args.resume

    if try_to_resume_training and os.path.exists(best_checkpoint):
        iter_num = train_utils.resume_training(args, best_filename, results_dir,
                                    model, optimizer, scheduler, best)

    train_loss = train_utils.AverageMeter()
    train_acc = train_utils.AverageMeter()

    # Here is our training loop.
    accum = args.batch_accum - 1
    while iter_num < max_iter:
        for t_input, t_target in train_loader:
            # Load data
            input_var, target_var = [d.cuda()
                    for d in [t_input, t_target]]
            # Evaluate model
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss.update(loss.data.item(), t_input.size(0))
            # Perform one step of SGD
            if iter_num > 0:
                if accum == args.batch_accum - 1:
                    optimizer.zero_grad()
                loss.backward()
                if accum == 0:
                    optimizer.step()
                    scheduler.step()
            # Check training set accuracy
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    t_input.size(0))
            train_acc.update(accuracy)
            if accum > 0:
                accum -= 1
                continue

            # Ocassionally check validation set accuracy and checkpoint
            keep = args.keep and (iter_num in snapshot_iters
                    or iter_num > 200 and (iter_num + 100 in snapshot_iters)
                    or iter_num == max_iter - 1)
            if keep or iter_num % 1000 == 0:
                train_utils.validate_and_checkpoint(keep, model, val_loader,
                                                    criterion, iter_num,
                                                    optimizer, scheduler,
                                                    train_acc, train_loss,
                                                    best, results_dir,
                                                    checkpoint_filename,
                                                    best_filename,
                                                    args.model,
                                                    args.w
                                                    )
                model.train()
                if iter_num % 1000 == 0:
                    train_loss.reset()
                    train_acc.reset()

            iter_num += 1
            accum = args.batch_accum - 1
            if iter_num >= max_iter:
                break

def apply_init(model, method):
    for n, p in model.named_parameters():
        if 'bias' in n:
            torch.nn.init.zeros_(p)
        elif len(p.shape) == 1:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.kaiming_normal_(p, nonlinearity='relu')

if __name__ == '__main__':
    def parseargs():
        parser = argparse.ArgumentParser()

        def aa(*args, **kwargs):
            parser.add_argument(*args, **kwargs)

        aa('--dataset', choices=['places', 'imagenet'], default='places')
        aa('--model', choices=['resnet18', 'resnet50'], default='resnet50')
        aa('--iters', type=int, default=131071)
        aa('--init_lr', type=float, default=1e-2)
        aa('--weight_decay', type=float, default=1e-4)
        aa('--ds_root', default='datasets')
        aa('--batch_size', type=int, default=32)
        aa('--batch_accum', type=int, default=2)
        aa('--init_method', choices=['k'], default='k')
        aa('--optimizer', choices=['adam', 'sgd'], default='adam')
        aa('--momentum', type=float, default=0.0)
        aa('--resume', type=int, default=0)
        aa('--keep', type=int, default=0)
        aa('--prefix', type=str, default="")
        aa('--w', nargs="+", type=int, default=[])
        args = parser.parse_args()
        return args
    args = parseargs()
    main(args)
