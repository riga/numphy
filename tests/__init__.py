# -*- coding: utf-8 -*-
# flake8: noqa


# adjust the path to import numphy
import os
import sys

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)


# import all tests
from .test_core import *
from .test_util import *
