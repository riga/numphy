# -*- coding: utf-8 -*-
# flake8: noqa

"""
Physics objects backed by NumPy and/or TensorFlow.
"""


__author__ = "Marcel Rieger"
__email__ = "python-numphy@googlegroups.com"
__credits__ = ["Benjamin Fischer", "Marcel Rieger"]
__copyright__ = "Copyright 2018, Marcel Rieger"
__contact__ = "https://github.com/riga/numphy"
__license__ = "MIT"
__status__ = "Development"
__version__ = "0.0.1"

__all__ = [
    "Wrapper", "Trace", "DataProxy",
    "is_numpy", "is_tensorflow", "map_struct", "is_lazy_iterable", "no_value",
    "t", "HAS_NUMPY", "HAS_TENSORFLOW",
]


# provisioning imports
from numphy.core import *
from numphy.util import *

