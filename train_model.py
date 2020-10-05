import pytorch_lightning as pl
import torch
import torch.hub
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
from src import *
from torchvision.models import *
import toml
import time

test = googlenet  # dummy

config = toml.load('config.toml')
config["DATASET_PATH"] = '/data/vision/torralba/datasets/places/files'
config["BATCH_SIZE"] = 200
config["MODEL_NAME"] = 'googlenet'
config["SAVE_DIR"] = 'results/'
config["MAX_EPOCHS"] = 4987896

train_loader, val_loader = get_loaders(f'{config["DATASET_PATH"]}/train', f'{config["DATASET_PATH"]}/val',
                                       batch_size=config["BATCH_SIZE"])

NUM_CLASSES = len(train_loader.dataset.classes)

MODEL_LOOKUP = {  # maps network key name to command which instantiates it (with NUM_CLASSES classes)
    "googlenet": "GoogLeNet(num_classes=NUM_CLASSES)",
}


class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # grab lr and any other hyperparams
        self.model = eval(MODEL_LOOKUP[model_name])
        self.iter = torch.LongTensor([0], device=self.device)

    def forward(self, x):
        """Forward fn for inference"""
        return F.softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)[0]  # .logits is GoogleLeNet specific?
        loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(F.softmax(y_hat, dim=1), dim=1)
        self.iter += 1
        result = pl.TrainResult(minimize=loss)
        result.log_dict({'train_loss': loss, 'train_accuracy': accuracy(y_pred, y)}, prog_bar=True)
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
                                                        steps_per_epoch=len(train_loader) // self.hparams.batch_size,
                                                        epochs=config["MAX_EPOCHS"])
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]


checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(config["SAVE_DIR"], config["MODEL_NAME"] + time.strftime("%Y%m%d-%H%M%S"),
                          '{iter:08.0f}-{val_accuracy:.2f}'),
    save_top_k=-1, period=0, save_weights_only=True
)

classifier = ImageClassifier(batch_size=config["BATCH_SIZE"], model_name=config["MODEL_NAME"],
                             lr=float("inf"))
wandb_logger = WandbLogger(name=f'{config["MODEL_NAME"]}', project='places_benchmark')
trainer = pl.Trainer(gpus=[0], callbacks=[ValidateCheckpointCallback()], checkpoint_callback=checkpoint_callback,
                     logger=wandb_logger,  # auto_lr_find='lr',
                     check_val_every_n_epoch=float("inf"))  # allow callback to control validation :)
trainer.fit(classifier, train_loader, val_loader)
