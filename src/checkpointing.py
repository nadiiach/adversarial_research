from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_model(self, filepath, trainer, pl_module):
        if trainer.current_epoch in snapshot_iters:
            super()._save_model(filepath, trainer, pl_module)