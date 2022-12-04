from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.variables.constants import onehot2label, label2str


class LoggerSingletonMeta(type):
    _instance: Optional[Logger] = None

    def __call__(self) -> Logger:
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class Logger(metaclass=LoggerSingletonMeta):
    def __init__(self, time_format="%d-%m-%Y_%H:%M:%S",
                 label_decoder=onehot2label):
        self.datetime = datetime.now().strftime(time_format)
        self.label_decoder = label_decoder

    def set_run_config(self, config):
        self.run_config = config
        self.workdir = os.path.join(self.run_config['logging_dir'],
                                    self.datetime)
        self.make_path(self.workdir)
        self.log_inputs = self.run_config['log_inputs']
        self.log_heatmaps = self.run_config['log_heatmaps']
        self.write_run_config(self.run_config)
        self.input_counters = [0, 0, 0, 0]
        self.log_freq = self.run_config['input_log_freq']

    def write_run_config(self, run_config):
        with open(os.path.join(self.workdir, 'run_config.yml'),
                  'w') as outfile:
            yaml.dump(run_config, outfile, default_flow_style=False)

    def make_path(self, path):
        os.makedirs(path, exist_ok=True)

    def log_input(self, X, y, prefix):
        if self.log_inputs:
            label_str = label2str(self.label_decoder(y))
            one_index = y.index(max(y))
            self.input_counters[one_index] += 1
            if self.input_counters[one_index] % self.log_freq == 0:
                dir_path = os.path.join(self.workdir, label_str)
                self.make_path(dir_path)
                img_path = os.path.join(dir_path, prefix + '_' + str(
                    self.input_counters[one_index]) + '.png')
                plt.imshow(np.squeeze(X, axis=2))
                plt.savefig(img_path, format='png', bbox_inches='tight')
                plt.close()
