from typing import TYPE_CHECKING

from ...utils import (
    _LazyModule
)


_import_structure = {
    "configuration_my_model": ["My_Config"],
    "modeling_my_model": ["My_Model_1"]
}



if TYPE_CHECKING:
    from .configuration_my_model import My_Config
    from .modeling_my_model import My_Model_1
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

