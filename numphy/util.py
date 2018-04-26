# -*- coding: utf-8 -*-

"""
Helpful utility functions.
"""


__all__ = ["is_numpy", "is_tensorflow", "is_lazy_iterable", "no_value", "cached_property"]


import types
import collections

import six


def is_numpy(obj):
    """
    Returns *True* when *obj* is a numpy object.
    """
    return type(obj).__module__.split(".") == "numpy"


def is_tensorflow(obj):
    """
    Returns *True* when *obj* is a tensorflow object.
    """
    return type(obj).__module__.split(".")[0] == "tensorflow"


def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))


class NoValue(object):

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False


#: Unique dummy value that evaluates to *False*.
no_value = NoValue()


class cached_property(property):
    """
    Version of Python's built-in property that also implements caching. Example:

    .. code-block:: python

        class MyClass(object):

            @cached_property
            def foo(self):
                print("computing foo ...")
                return some_heavy_computation()  # 27

            @foo.setter
            def foo(self, value):
                # no need to set the internal, cached member
                # just return the value
                return value

        c = MyClass()

        c.foo
        # "computing foo ..."
        # -> 27

        c.foo
        # -> 27

        c.foo = 42
        c.foo
        # -> 42

        del c.foo
        c.foo
        # "computing foo ..."
        # -> 27
    """

    def __init__(self, *args, **kwargs):
        super(cached_property, self).__init__(*args, **kwargs)

        # store the cache attribute
        self.cache_attr = "_{}".format(self.fget.__name__) if self.fget else None

    def __get__(self, obj, objtype=None):
        # compute and set the cache value only once
        if getattr(obj, self.cache_attr, no_value) == no_value:
            setattr(obj, self.cache_attr, super(cached_property, self).__get__(obj, objtype))
        return getattr(obj, self.cache_attr)

    def __set__(self, obj, value):
        # when no setter is defined, let the super class raise the exception
        # otherwise, call fset and cache its return value
        if not self.fset:
            super(cached_property, self).__set__(obj, value)
        elif self.cache_attr:
            setattr(obj, self.cache_attr, self.fset(obj, value))

    def __delete__(self, obj):
        # when the deleter was set explicitly, use it
        # otherwise, delete the cache value
        if self.fdel:
            super(cached_property, self).__delete__(obj)
        elif self.cache_attr:
            delattr(obj, self.cache_attr)
