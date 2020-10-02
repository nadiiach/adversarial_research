from torchvision.datasets import ImageFolder
from torchvision.models import squeezenet1_1
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.hub
from src import *

DATASET_PATH = '/data/vision/torralba/datasets/places/files'
BATCH_SIZE = 32

train_loader, val_loader = get_loaders(f'{DATASET_PATH}/train', f'{DATASET_PATH}/val', batch_size=BATCH_SIZE)

NUM_CLASSES = len(train_loader.dataset.classes)

from torchvision.models import *

MODEL_LOOKUP = {
    "googlenet": "GoogLeNet(num_classes=NUM_CLASSES)",
}


class ImageClassifier(pl.LightningModule):
    def __init__(self, lr, batch_size, model_name):
        super().__init__()
        self.save_hyperparameters()  # grab lr and any other hyperparams
        self.model = eval(MODEL_LOOKUP[model_name])

    def forward(self, x):
        """Forward fn for inference"""
        return F.softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(F.softmax(y_hat), dim=1)

        result = pl.TrainResult(minimize=loss)
        result.log_dict({'train_loss': loss, 'train_accuracy': accuracy(y_pred, y)})
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(F.softmax(y_hat), dim=1)

        result = pl.EvalResult()
        result.log_dict({'val_loss': loss, 'val_accuracy': accuracy(y_pred, y)})
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.hparams.lr,
                                                        steps_per_epoch=len(train_loader) / self.hparams.batch_size,
                                                        epochs=self.hparams.epochs)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath='my/path/{epoch}-{val_loss:.2f}-{other_metric:.2f}',
    save_top_k=-1,
    save_last=True,
)

classifier = ImageClassifier(lr=1, batch_size=32, model_name='googlenet')
trainer = pl.Trainer()
trainer.fit(classifier, train_loader, val_loader)
