import json
import threading
import time

import psutil
import torch
import numpy as np
import random
import torch.optim as optim
import pickle as pkl

def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    optimizer = None
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
    return optimizer


def random_shuffle(x, y):
    idx = list(range(len(x)))
    random.shuffle(idx)
    return x[idx], y[idx]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
# -*- coding: utf-8 -*-

def load_metadata(path):
    with open('../data/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_graph_dataset(path, t, n):

    with open('../data/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data


class MemoryMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.memory_usage = []
        self.monitoring = False

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self.record_memory_usage)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        self.thread.join()

    def record_memory_usage(self):
        process = psutil.Process()
        while self.monitoring:
            memory_info = process.memory_info()
            self.memory_usage.append(memory_info.rss)  # RSS memory in bytes
            time.sleep(self.interval)

    def get_average_memory_usage(self):
        if not self.memory_usage:
            return 0
        return sum(self.memory_usage) / len(self.memory_usage)

    def get_peak_memory_usage(self):
        if not self.memory_usage:
            return 0
        return max(self.memory_usage)




