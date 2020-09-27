# Training program for alexnet.

import torch, os, shutil, torchvision, argparse
from net.models import flexresnet
from netdissect import parallelfolder, renormalize, pbar
from torchvision import transforms

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
   5931642, 7053950, 8388608, 9975792, 11863283, 14107901, 16777216,
   19951585, 23726566, 28215802
]

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--dataset', choices=['places', 'imagenet'], default='places')
    aa('--arch', choices=['resnet18', 'resnet50'], default='resnet50')
    aa('--iters', type=int, default=snapshot_iters[-1])
    aa('--init_lr', type=float, default=1e-3)
    aa('--weight_decay', type=float, default=0.0)  # default=1e-4)
    aa('--ds_root', default='datasets')
    aa('--batch_size', type=int, default=32)
    aa('--batch_accum', type=int, default=2)
    aa('--init_method', choices=['k'], default='k')
    aa('--network_width', type=int, default=64)
    aa('--fc_batchnorm', type=int, default=0)
    aa('--optimizer', choices=['adam', 'sgd'], default='adam')
    aa('--momentum', type=float, default=0.0)
    aa('--resume', type=int, default=0)
    aa('--keep', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    results_dir = 'results/%s/%s%s%s-%s-%2g-b%d-i%s-w%g-%s-m%g-it%d' % (
            args.arch,
            '' if args.network_width == 64 else f'nw{args.network_width}-',
            '' if args.fc_batchnorm == 0 else 'bnf-',
            args.dataset, args.optimizer, args.init_lr, args.batch_size * args.batch_accum,
            args.init_method,
            args.weight_decay, args.optimizer, args.momentum,
            args.iters)
    training_dir = '%s/%s/train' % (args.ds_root, args.dataset)
    val_dir = '%s/%s/val' % (args.ds_root, args.dataset)
    num_classes = dict(places=365, imagenet=1000)[args.dataset]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'args.txt'), 'w') as f:
        f.write(str(args) + '\n')
    def printstat(s):
        with open(os.path.join(results_dir, 'log.txt'), 'a') as f:
            f.write(str(s) + '\n')
        pbar.print(s)
    printstat(str(args))
    # Here's our data
    train_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([training_dir],
            classification=True,
            transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=48, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([val_dir],
            classification=True,
            transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        batch_size=64, shuffle=False,
        num_workers=24, pin_memory=True)
    # We initialize the model with some random, some identity convolutions.
    # TODO: consider setting include_lrn=False here. (Left it on by accident.)
    if args.arch == 'resnet50':
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.arch == 'resnet18':
        if args.network_width == 64 and not args.fc_batchnorm:
            model = torchvision.models.resnet18(num_classes=num_classes)
        else:
            model = flexresnet.resnet18(num_classes=num_classes, network_width=args.network_width,
                    fc_batchnorm=args.fc_batchnorm)

    apply_init(model, args.init_method)
    model.train()
    model.cuda()

    max_iter = args.iters + 1
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                lr=args.init_lr,
                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                lr=args.init_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
    iter_num = 0
    best = dict(val_accuracy=0.0)
    # Oh, hold on.  Let's actually resume training if we already have a model.
    checkpoint_filename = 'weights.pth'
    best_filename = 'best_%s' % checkpoint_filename
    best_checkpoint = os.path.join(results_dir, best_filename)
    try_to_resume_training = args.resume
    if try_to_resume_training:
        assert os.path.exists(os.path.join(results_dir, checkpoint_filename))
        checkpoint = torch.load(os.path.join(results_dir, checkpoint_filename))
        iter_num = checkpoint['iter'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best['val_accuracy'] = checkpoint['accuracy']
        print(f'Resuming from iteration {iter_num}')

    # Track the average training loss/accuracy between snapshots.
    train_loss, train_acc = AverageMeter(), AverageMeter()

    def save_checkpoint(state, is_best, iter_copy):
        filename = os.path.join(results_dir, checkpoint_filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename,
                    os.path.join(results_dir, best_filename))
        if iter_copy is not None:
            iter_filename = 'iter_%d_weights.pth' % iter_copy
            shutil.copyfile(filename,
                    os.path.join(results_dir, iter_filename))

    def validate_and_checkpoint(keep_snapshot):
        model.eval()
        val_loss, val_acc = AverageMeter(), AverageMeter()
        for input, target in pbar(val_loader):
            # Load data
            input_var, target_var = [d.cuda() for d in [input, target]]
            # Evaluate model
            with torch.no_grad():
                output = model(input_var)
                loss = criterion(output, target_var)
                _, pred = output.max(1)
                accuracy = (target_var.eq(pred)
                        ).data.float().sum().item() / input.size(0)
            val_loss.update(loss.data.item(), input.size(0))
            val_acc.update(accuracy, input.size(0))
            # Check accuracy
            pbar.post(l=val_loss.avg, a=val_acc.avg)
        # Save checkpoint
        save_checkpoint({
            'iter': iter_num,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'accuracy': val_acc.avg,
            'loss': val_loss.avg,
            'train_accuracy': train_acc.avg,
            'train_loss': train_loss.avg,
          },
          val_acc.avg > best['val_accuracy'],
          iter_num if keep_snapshot else None)
        best['val_accuracy'] = max(val_acc.avg, best['val_accuracy'])
        printstat(
            'Iteration %d val accuracy %.2f loss %.2f' %
                (iter_num, val_acc.avg * 100.0, val_loss.avg)
            + ' train accuracy %.2f loss %.2f' %
                (train_acc.avg * 100.0, train_loss.avg)
        )

    # Here is our training loop.
    accum = args.batch_accum - 1
    while iter_num < max_iter:
        for t_input, t_target in pbar(train_loader):
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
            # Check training set accuracy
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    t_input.size(0))
            train_acc.update(accuracy)
            if accum > 0:
                accum -= 1
                continue
            pbar.post(l=train_loss.avg, a=train_acc.avg,
                    v=best['val_accuracy'])
            # Ocassionally check validation set accuracy and checkpoint
            keep = args.keep and (iter_num in snapshot_iters
                    or iter_num > 200 and (iter_num + 100 in snapshot_iters)
                    or iter_num == max_iter - 1)
            if keep or iter_num % 1000 == 0:
                validate_and_checkpoint(keep)
                model.train()
                if iter_num % 1000 == 0:
                    train_loss.reset()
                    train_acc.reset()
            # Advance
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
