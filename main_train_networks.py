import argparse
from net.models import train_alexnet, train_vgg, train_resnet



def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa('--dataset', choices=['places', 'imagenet'], default='places')
    aa('--iters', type=int, default=4987896)
    aa('--init_lr', type=float, default=2e-2)
    aa('--weight_decay', type=float, default=5e-4)
    aa('--ds_root', default='datasets')
    aa('--batch_size', type=int, default=256)
    aa('--batch_accum', type=int, default=4)
    aa('--init_method', choices=['k'], default='k')
    aa('--optimizer', choices=['adam', 'sgd'], default='sgd')
    aa('--momentum', type=float, default=0.9)
    aa('--dropout', type=int, choices=[0, 1], default=1)
    aa('--groups', type=int, choices=[0, 1], default=1)
    aa('--keep', type=int, default=1)
    aa('--resume', type=str, default="last", choices=['best', 'last'])
    aa('--model', choices=['alexnet', 'vgg', 'resnet18',
                                            'resnet50'], required=True)
    aa('--prefix', type=str, default="")
    aa('--w', nargs="+", type=int, default=[])
    args = parser.parse_args()
    return args


args = parseargs()
if "alex" in args.model:
    train_alexnet.main(args)

if "vgg" in args.model:
    train_vgg.main(args)

if "resnet" in args.model:
    train_resnet.main(args)



