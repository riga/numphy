# -*- coding: utf-8 -*-

"""
Helpful utility functions.
"""


__all__ = ["empty_slice", "is_lazy_iterable", "expand_ellipsis", "no_value"]


import types
import collections

import six


empty_slice = slice(None)


def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))


def expand_ellipsis(values, size):
    if Ellipsis not in values:
        return values

    n = size - len(values) + 1
    if n <= 0:
        raise Exception("size {} not sufficient to expand ellipsis".format(size))

    idx = values.index(Ellipsis)
    return values[:idx] + n * (empty_slice,) + values[idx + 1:]


class NoValue(object):

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False


#: Unique dummy value that evaluates to *False*.
no_value = NoValue()
