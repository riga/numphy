# -*- coding: utf-8 -*-

"""
Physics objects backed by NumPy and/or TensorFlow.
"""


__author__ = "Marcel Rieger"
__email__ = "python-numphy@googlegroups.com"
__copyright__ = "Copyright 2018, Marcel Rieger"
__contact__ = "https://github.com/riga/numphy"
__license__ = "MIT"
__status__ = "Development"
__version__ = "0.0.1"

__all__ = ["Data", "Trace", "t"]


from numphy.util import no_value, is_lazy_iterable

import six


class Trace(object):

    @classmethod
    def make_tuple(cls, trace):
        return trace if isinstance(trace, tuple) else (trace,)

    @classmethod
    def get(cls, trace, struct):
        if not trace:
            return struct

        # ensure that trace is a tuple
        trace = trace.trace if isinstance(trace, cls) else cls.make_tuple(trace)

        first, rest = trace[0], trace[1:]
        return cls.get(rest, struct[first])

    @classmethod
    def set(cls, trace, struct, value):
        if not trace:
            return value

        # ensure that trace is a tuple
        trace = trace.trace if isinstance(trace, cls) else cls.make_tuple(trace)

        first, rest = trace[0], trace[1:]
        struct[first] = cls.set(rest, struct[first], value)

        return struct

    def __init__(self, trace):
        super(Trace, self).__init__()

        # set the trace
        self._trace = tuple()
        self.trace = trace

    def __repr__(self):
        return "<{} {} at {}>".format(self.__class__.__name__, str(self), hex(id(self)))

    def __str__(self):
        return self.trace.__str__()

    def __len__(self):
        return self.trace.__len__()

    def __nonzero__(self):
        return len(self) > 0

    def __getitem__(self, key):
        item = self.trace.__getitem__(key)
        return item if isinstance(key, six.integer_types) else self.__class__(item)

    def __setitem__(self, key, value):
        trace = list(self.trace)

        # mimic list behavior
        try:
            trace.__setitem__(key, value)
        except IndexError:
            raise IndexError("index {} out of range for trace {!r}".format(key, self))
        except TypeError:
            raise TypeError("trace indices must be integers, not {}".format(type(key).__name__))

        self.trace = tuple(trace)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        return self.__class__(self.trace + other.trace)

    def __call__(self, struct, value=no_value):
        if value is no_value:
            return self.get(self.trace, struct)
        else:
            return self.set(self.trace, struct, value)

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, trace):
        if isinstance(trace, self.__class__):
            trace = trace.trace

        self._trace = self.make_tuple(trace)


class _TraceFactory(object):

    def __getitem__(self, trace):
        return trace if isinstance(trace, Trace) else Trace(trace)


t = _TraceFactory()

import numpy as np

# events = np.load("dy.npy")
# print events["Muon_Px"][2]
# lves = LorentzVectors(events, )

