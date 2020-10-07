import pytorch_lightning as pl
import torch
import torch.hub
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
from src import *
from torchvision.models import *

_ = googlenet  # for pycharm, gets rid of unused import warning
import toml
import time
import argparse

# Get Runtime Arguments (general configurations)
parser = argparse.ArgumentParser(prog="Train Model", description="Train places/ImageNet benchmarks on various models.")
parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int)
parser.add_argument('-c', '--config', required=True, dest='config_path', type=str)
parser.add_argument('-d', '--dataset', default='places', dest='dataset', type=str, help='(places, imagenet, etc.)')

args = parser.parse_args()

config = toml.load(args.config_path)

save_folder = f'{config["model"]["name"]}_{args.dataset}_{time.strftime("%Y-%m-%d_%H%M%S")}'
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(config["paths"]["save"], save_folder, '{iter:08.0f}-{val_accuracy:.2f}'),
    save_top_k=-1, period=0,
)
wandb_logger = WandbLogger(name=f'{config["model"]["name"]}', project=f'{args.dataset}_benchmark')
wandb_logger.experiment.log({'checkpoint_path': os.path.join(config["paths"]["save"], save_folder)})

train_loader, val_loader = get_loaders(f'{DATASET_PATHS[args.dataset]}/train', f'{DATASET_PATHS[args.dataset]}/val',
                                       batch_size=config["training"]["batch_size"])

NUM_CLASSES = len(train_loader.dataset.classes)  # NUM_CLASSES hook for config files to use
NUM_BATCHES = len(train_loader)  # batches per epoch


class ImageClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = eval(config['model']['instantiate'])
        self.iter = torch.LongTensor([0], device=self.device)

    def forward(self, x):
        """Forward fn for inference"""
        return F.softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        if 'aux_weights' in config['model']:  # aggregate loss if aux classifiers present
            y_hat = self.model(x)
            loss = sum(F.cross_entropy(aux_out, y) * aux_weight
                       for aux_out, aux_weight in zip(y_hat, config['model']['aux_weights']))
            y_hat = y_hat[0]
        else:
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(F.softmax(y_hat, dim=1), dim=1)
        result = pl.TrainResult(minimize=loss)
        result.log_dict({'train_loss': loss, 'train_accuracy': accuracy(y_pred, y)}, on_epoch=True)
        self.iter += 1
        return result

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(F.softmax(y_hat, dim=1), dim=1)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_loss': loss, 'val_accuracy': accuracy(y_pred, y), 'iter': self.iter})
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.hparams.lr,
                                                        steps_per_epoch=NUM_BATCHES,
                                                        epochs=snapshot_iters[-1] // NUM_BATCHES)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


classifier = ImageClassifier(batch_size=config["training"]["batch_size"], model_name=config["model"]["name"],
                             lr=float("inf"))

trainer = pl.Trainer(gpus=[args.gpu], callbacks=[ValidateCheckpointCallback()], checkpoint_callback=checkpoint_callback,
                     logger=wandb_logger, auto_lr_find='lr',
                     check_val_every_n_epoch=float("inf"), )  # allow callback to control validation :)

trainer.fit(classifier, train_loader, val_loader)
