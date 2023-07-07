from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn


from ...modeling_utils import PreTrainedModel
from ...utils import (
    logging,
)
from .configuration_my_model import My_Config


logger = logging.get_logger(__name__)

class My_Model_1(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        input_size = config.input_size
        hidden_size = config.hidden_size
        output_size = config.output_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x