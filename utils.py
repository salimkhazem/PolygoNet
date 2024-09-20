import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AverageMeter:
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


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = logdir / run_name
        if not log_path.exists():
            log_path.mkdir(parents=True)
            return log_path
        i = i + 1


def get_optimizer(cfg, model):
    optimizer_cls = cfg["Optimizer"]["Name"]
    optimizer_args = {k: v for k, v in cfg["Optimizer"].items() if k != "Name"}

    return eval(f"optim.{optimizer_cls}")(model.parameters(), **optimizer_args)


def get_criterion(cfg):
    if cfg["Loss"]["Name"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif cfg["Loss"]["Name"] == "MSECE": 
        criterion_coord = nn.MSELoss() 
        criterion_mask = nn.CrossEntropyLoss() 
        return criterion_mask, criterion_coord


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
