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

def ellipsis_eqiv(key):
    """
    Returns whether *key* is a *tuple* of *Ellipsis* or equivalent *slice*s.
    """
    return isinstance(key, tuple) and all(
        k is Ellipsis or
        isinstance(k, slice) and (
            k.start in (None, 0) and
            k.stop in (None, -1) and
            k.step in (None, 1)
        )
        for k in key
    )

def slice_len(sli, ref=None):
    """
    Returns the expected length of the slice *sli*, possibly consulting the reference length *ref*
    or *len(ref)* if the former is not an *int*. Will return *None* if a *ref* is needed but *None*.
    """
    b, e, s = sli.start, sli.stop, sli.step
    if s is None:
        s = 1
    if (0 <= b and 0 <= e) or (b < 0 and e < 0):
        return len(xrange(b, e, abs(s)))
    elif ref is not None:
        if not isinstance(ref, int):
            ref = len(ref)
        return len(xrange(*sli.indices(ref)))


class NoValue(object):

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False


#: Unique dummy value that evaluates to *False*.
no_value = NoValue()
