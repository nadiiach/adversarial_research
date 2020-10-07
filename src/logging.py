import wandb


class CumulativeWandbLogger:
    def __init__(self, name=None, project=None, ):
        if name is not None and project is not None:
            wandb.init(name=name, project=project)
        self.count = 0
        self.metrics = {}

    def add_metrics(self, metrics):
        for metric in metrics:
            self.metrics[metric] = self.metrics.get(metric, 0) + metrics[metric]
        self.count += 1

    def aggregate_and_log(self):
        wandb.log({k: v / self.count for k, v in self.metrics.items()})
        self.count = 0
        self.metrics = {}
