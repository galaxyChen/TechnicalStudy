from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class My_Config(PretrainedConfig):

    model_type = "my_model"
    def __init__(
        self,
        input_size=10,
        hidden_size=20,
        output_size=5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size