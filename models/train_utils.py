import os, torch, shutil
from netdissect import parallelfolder, renormalize, pbar
from torchvision import transforms

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


def printstat(s, results_dir):
    with open(os.path.join(results_dir, 'log.txt'), 'a') as f:
        f.write(str(s) + '\n')
    pbar.print(s)


def save_checkpoint(state, is_best, iter_copy, results_dir,
                    checkpoint_filename, best_filename):
    filename = os.path.join(results_dir, checkpoint_filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(results_dir, best_filename))
        print("Saved iter_filename={}".format(
                os.path.join(results_dir, best_filename)))
        if iter_copy is not None:
            best_filename_2 = 'best_iter_%d_weights.pth' % iter_copy
            shutil.copyfile(filename,os.path.join(results_dir, best_filename_2))
            print("Saved iter_filename={}".format(
                os.path.join(results_dir, best_filename_2)))

    if iter_copy is not None:
        iter_filename = 'iter_%d_weights.pth' % iter_copy
        shutil.copyfile(filename, os.path.join(results_dir, iter_filename))
        print("Saved iter_filename={}".format(
            os.path.join(results_dir, iter_filename)))

def validate_and_checkpoint(keep_snapshot, model, val_loader, criterion,
                            iter_num, optimizer, scheduler, train_acc,
                            train_loss, best, results_dir,
                            checkpoint_filename, best_filename):
    model.eval()
    val_loss, val_acc = AverageMeter(), AverageMeter()
    for input, target in val_loader:
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
        'scheduler' : scheduler.state_dict(),
        'accuracy': val_acc.avg,
        'loss': val_loss.avg,
        'train_accuracy': train_acc.avg,
        'train_loss': train_loss.avg,
      },
        val_acc.avg > best['val_accuracy'],
        iter_num if keep_snapshot else None,
        results_dir,
        checkpoint_filename,
        best_filename
    )

    best['val_accuracy'] = max(val_acc.avg, best['val_accuracy'])
    printstat(
        'Iteration %d val accuracy %.2f loss %.2f' %
            (iter_num, val_acc.avg * 100.0, val_loss.avg)
        + ' train accuracy %.2f loss %.2f' %
            (train_acc.avg * 100.0, train_loss.avg), results_dir
    )

def get_loders(resize, crop, train_workers, val_workers,
               training_dir, val_dir, args):
    train_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([training_dir],
            classification=True,
            transform=transforms.Compose([
                        transforms.Resize(resize),
                        transforms.RandomCrop(crop),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=train_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([val_dir],
            classification=True,
            transform=transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(crop),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        #batch_size=64, shuffle=False,
        batch_size=args.batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True)

    return train_loader, val_loader

def get_crit_opt_sched(model, max_iter, args):
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

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
            args.init_lr, max_iter - 1)
    return criterion, optimizer, scheduler

def resume_training(args, best_filename, results_dir,
                    model, optimizer, scheduler, best):
    print("Resuming training from iter {}".format(args.resume))

    if args.resume == "best":
        resume_filename = best_filename
        print("Resuming best iter {}".format(resume_filename))
    elif args.resume == "last":
        last_iter = get_last_iter(results_dir)
        resume_filename = 'iter_%s_weights.pth' % last_iter
        print("Resuming last iter {}".format(resume_filename))

    checkpoint = torch.load(os.path.join(results_dir, resume_filename))
    iter_num = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best['val_accuracy'] = checkpoint['accuracy'] # is this correct>
    print("Finished resuming training!")
    print("Resumed iter_num={}".format(iter_num))
    return iter_num

def get_last_iter(results_dir):
    files = os.listdir(results_dir)
    files = [x for x in files if "iter_" in x]
    try:
        iters = []
        for it in files:
            x = int(it.split("_")[1]) if "best" not in it \
                else int(it.split("_")[2])
            iters.append(x)

    except Exception as e:
        print("Exception {}".format(e))
        print("x {}".format(x))
        exit()

    return max(iters)

