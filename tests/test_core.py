# -*- coding: utf-8 -*-


__all__ = ["CoreFuncTest", "TraceTest", "DataProxyTest", "WrapperTest"]


import unittest
from copy import deepcopy

import numphy as ph


if ph.HAS_NUMPY:
    import numpy as np
else:
    np = None


class CoreFuncTest(unittest.TestCase):

    if np:
        def test_is_numpy(self):
            arr = np.arange(10)
            self.assertTrue(ph.is_numpy(arr))

    if ph.HAS_TENSORFLOW:
        def test_is_tensorflow(self):
            import tensorflow as tf

            t = tf.constant(range(10), tf.int32)
            self.assertTrue(ph.is_tensorflow(t))

    def test_map_struct(self):
        func = lambda value: value * 2

        # list
        struct = [1, 2, 3]
        self.assertIsInstance(ph.map_struct(func, struct), list)
        self.assertEqual(tuple(ph.map_struct(func, struct)), (2, 4, 6))

        # tuple
        struct = (1, 2, 3)
        self.assertEqual(ph.map_struct(func, struct), (2, 4, 6))

        # dict
        struct = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(tuple(sorted(ph.map_struct(func, struct).items())),
            (("a", 2), ("b", 4), ("c", 6)))

        # unevenly nested object
        struct = [[1, 2], 3, 4, {"a": 5, "b": 6}]
        result = ph.map_struct(func, struct)
        self.assertEqual(tuple(result[0]), (2, 4))
        self.assertEqual(tuple(result[1:3]), (6, 8))
        self.assertEqual(tuple(sorted(result[3].items())), (("a", 10), ("b", 12)))

        # numpy objects
        if np:
            # structured array
            struct = np.array([(1, 2.), (3, 4.)], dtype=[("f0", "i4"), ("f1", "f4")])
            result = ph.map_struct(func, struct)
            self.assertEqual(tuple(result["f0"]), (2, 6))
            self.assertEqual(tuple(result["f1"]), (4., 8.))

            # structured array with object fields
            struct = np.array([(1, struct)], dtype=[("f0", "i4"), ("f1", "O")])
            result = ph.map_struct(func, struct)
            self.assertEqual(result[0][0], 2)
            self.assertEqual(tuple(result[0][1].tolist()), ((2, 4.), (6, 8.)))


class TraceTest(unittest.TestCase):

    struct = {
        "a": {
            1: list(range(10, 20)),
        },
    }

    if np:
        np_struct = {
            "a": {
                1: np.arange(100).reshape(10, 10),
            },
        }

    trace = ph.t["a", 1, 0]

    def test_get(self):
        self.assertEqual(ph.Trace.get(self.trace, self.struct), 10)

        if np:
            self.assertEqual(tuple(ph.Trace.get(self.trace, self.np_struct)), tuple(range(10)))

    def test_set(self):
        struct = deepcopy(self.struct)
        ph.Trace.set(self.trace, struct, 99)
        self.assertEqual(ph.Trace.get(self.trace, struct), 99)

        if np:
            np_struct = deepcopy(self.np_struct)
            ph.Trace.set(self.trace, np_struct, np.arange(30, 40))
            self.assertEqual(tuple(ph.Trace.get(self.trace, np_struct)), tuple(range(30, 40)))

    def test_make_tuple(self):
        self.assertEqual(ph.Trace._make_tuple(1), (1,))
        self.assertEqual(ph.Trace._make_tuple([1, 2]), ([1, 2],))
        self.assertEqual(ph.Trace._make_tuple((1, 2)), (1, 2))

    def test_init(self):
        trace = ph.Trace()
        self.assertEqual(trace, ())

        trace = ph.Trace(1)
        self.assertEqual(trace, (1,))

        trace = ph.Trace([1, 2])
        self.assertEqual(trace, ([1, 2],))

        trace = ph.Trace(trace)
        self.assertEqual(trace, ([1, 2],))

    def test_magic_methods(self):
        self.assertEqual(str(self.trace), "('a', 1, 0)")
        self.assertEqual(len(self.trace), 3)
        self.assertTrue(self.trace)
        self.assertFalse(ph.Trace())
        self.assertTrue("a" in self.trace)
        self.assertTrue(1 in self.trace)
        self.assertFalse(2 in self.trace)

    def test_get_item(self):
        self.assertEqual(self.trace[0], "a")
        self.assertEqual(self.trace[1], 1)
        self.assertIsInstance(self.trace[1:], ph.Trace)
        self.assertEqual(self.trace[1:].values, (1, 0))
        self.assertEqual(self.trace[:-1].values, ("a", 1))

    def test_set_item(self):
        trace = ph.Trace(self.trace)

        trace[0] = "b"
        self.assertEqual(trace[0], "b")

        trace[1:] = (2, 3)
        self.assertEqual(trace[1:], (2, 3))

    def test_add(self):
        trace = self.trace + (2, 3)
        self.assertEqual(trace, ("a", 1, 0, 2, 3))

        trace += (4,)
        self.assertEqual(trace, ("a", 1, 0, 2, 3, 4))

    def test_call(self):
        self.assertEqual(self.trace(self.struct), 10)

        struct = deepcopy(self.struct)
        self.trace(struct, 99)
        self.assertEqual(self.trace(struct), 99)

    def test_trace_factory(self):
        trace = ph.t[1, 2, 3]
        self.assertEqual(trace, (1, 2, 3))

        trace = ph.t[1, :]
        self.assertEqual(trace, (1, slice(None)))


class DataProxyTest(unittest.TestCase):

    def test_attributes(self):
        obj = list(range(10))
        proxy = ph.DataProxy(obj)

        self.assertIsNot(proxy, obj)
        self.assertEqual(len(proxy), len(obj))
        self.assertIn(0, proxy)
        self.assertEqual(proxy[0], obj[0])
        self.assertEqual(proxy.index(5), 5)

    def test_call(self):
        obj = list(range(10))
        proxy = ph.DataProxy(obj)
        self.assertIs(proxy(), obj)

        proxy(range(20, 40))
        self.assertEqual(len(proxy), 20)
        self.assertIsNot(proxy(), obj)


class WrapperTest(unittest.TestCase):

    @classmethod
    def wrapper(cls):
        return ph.Wrapper(data=deepcopy(TraceTest.struct), attrs=dict(
            a=ph.Wrapper(trace="a", attrs=dict(first=1))
        ))

    def test_get_data(self):
        tpl = (1, 2)
        w = ph.Wrapper(tpl)
        self.assertEqual(ph.Wrapper._get_data(w), tpl)
        self.assertEqual(ph.Wrapper._get_data(tpl), tpl)

    def test_init(self):
        w = self.wrapper()

        self.assertIsInstance(w.data_proxy(), dict)
        self.assertFalse(w.trace)
        self.assertEqual(len(w._attributes), 1)

        self.assertTrue(hasattr(w, "a"))
        self.assertFalse(hasattr(w, "b"))
        with self.assertRaises(AttributeError):
            w.b

        w = ph.Wrapper(trace=1, attrs=dict(
            foo=ph.Wrapper(trace=2),
            bar=3,
        ))
        self.assertEqual(w.trace.values, (1,))
        self.assertEqual(len(w._attributes), 2)
        self.assertIsInstance(w.foo, ph.Wrapper)
        self.assertEqual(w.foo.trace, (1, 2))
        self.assertIsInstance(w.bar, ph.Wrapper)
        self.assertEqual(w.bar.trace, (1, 3))

    def test_properties(self):
        w = self.wrapper()

        # data proxy getter
        self.assertIsInstance(w.data_proxy(), dict)
        self.assertIn("a", w.data_proxy)

        # data getter and setter
        w.a.data = list(range(5))
        self.assertEqual(w.a.first.data, 1)

        # trace getter and setter
        self.assertFalse(w.trace)
        self.assertEqual(w.a.first.trace, ("a", 1))
        w.a.first.trace[-1] = 2
        self.assertEqual(w.a.first.data, 2)

    def test_update_trace(self):
        w = self.wrapper()

        self.assertFalse(w.trace)
        self.assertEqual(w.a.trace, ("a",))
        self.assertEqual(w.a.first.trace, ("a", 1))

        update = lambda trace: (0,) + trace
        w.update_trace(update)

        self.assertEqual(w.trace, (0,))
        self.assertEqual(w.a.trace, (0, "a"))
        self.assertEqual(w.a.first.trace, (0, "a", 1))

    def test_magic_methods(self):
        w = self.wrapper()

        # __call__
        data = deepcopy(TraceTest.struct)
        data["a"][1][0] = 99
        self.assertEqual(w.a.first()[0], 10)
        w(data)
        self.assertEqual(w.a.first()[0], 99)

        # __repr__
        self.assertEqual(repr(w), repr(w()))

        # __contains__
        self.assertIn(15, w.a.first)

        # __nonzero__
        self.assertTrue(w)
        self.assertFalse(ph.Wrapper())

        # __len__
        self.assertEqual(len(w.a), 1)

    def test_get_item(self):
        w = self.wrapper()

        self.assertIsInstance(w["a"], ph.Wrapper)
        self.assertIsNot(w["a"], w)
        self.assertIsInstance(w.a.first[:5], ph.Wrapper)

        with self.assertRaises(Exception):
            w["a", 1]

    def test_set_item(self):
        w = self.wrapper()

        w.a = list(range(5))
        self.assertEqual(w.a.first.data, 1)

        w.a = w.a
        self.assertEqual(w.a.first.data, 1)

        with self.assertRaises(AttributeError):
            w.b = list(range(5))

    def test_wrap(self):
        w = self.wrapper()

        self.assertEqual(w.a.first()[0], 10)

        data = deepcopy(TraceTest.struct)
        data["a"][1][0] = 99

        w.wrap(data)
        self.assertIs(w.data_proxy(), data)
        self.assertEqual(w.a.first()[0], 10)

        w.wrap(data, overwrite=True)
        self.assertEqual(w.a.first()[0], 99)

    def test_sub(self):
        w = self.wrapper()
        w()[2] = "foo"

        w.sub(2, attr="foo", attrs=dict(o=-1))
        self.assertTrue(hasattr(w, "foo"))
        self.assertEqual(w.foo, "foo")
        self.assertEqual(w.foo.o, "o")

        back = w.a.first.sub(-1)
        self.assertFalse(hasattr(w.a.first, "back"))
        self.assertEqual(back(), 19)

        class MyWrapper(ph.Wrapper):

            pass

        w.sub(1, attr="other_a", cls=MyWrapper)
        self.assertIsInstance(w.other_a, MyWrapper)

    def test_copy(self):
        w = self.wrapper()

        # test shallow copy for object without trace
        shallow_copy = w.copy(deep=False)
        self.assertIsNot(w, shallow_copy)
        self.assertIs(w(), shallow_copy())
        self.assertEqual(w.trace, shallow_copy.trace)

        # test deep copy for object without trace
        deep_copy = w.copy(deep=True)
        self.assertIsNot(w, deep_copy)
        self.assertIsNot(w(), deep_copy())
        self.assertEqual(w.trace, deep_copy.trace)

        # test shallow copy for object with trace
        shallow_copy_a = w.a.copy(deep=False)
        self.assertIsNot(w.a, shallow_copy_a)
        self.assertIs(w.a(), shallow_copy_a())
        self.assertEqual(w.a.trace, shallow_copy_a.trace)

        # test deep copy for object with trace
        deep_copy_a = w.a.copy(deep=True)
        self.assertIsNot(w.a, deep_copy_a)
        self.assertIsNot(w.a(), deep_copy_a())
        self.assertNotEqual(w.a.trace, deep_copy_a.trace)

        # test deepcopy hook
        deep_copy = deepcopy(w)
        self.assertIsNot(w, deep_copy)
        self.assertIsNot(w(), deep_copy())
        self.assertEqual(w.trace, deep_copy.trace)

    def test_math_ops(self):
        w = ph.Wrapper((1, 2))

        self.assertEqual(w, w)
        self.assertNotEqual(w, ph.Wrapper((2, 3)))

        if np:
            w_np1 = ph.Wrapper(np.arange(0, 20, 2))
            w_np2 = ph.Wrapper(np.arange(5, 15))

            self.assertEqual((w_np1 < w_np2).sum(), 5)
            self.assertEqual((w_np1 <= w_np2).sum(), 6)
            self.assertEqual((w_np1 > w_np2).sum(), 4)
            self.assertEqual((w_np1 >= w_np2).sum(), 5)

        def W(start=-2, stop=2):
            return ph.Wrapper(tuple(range(start, stop)))

        w = W(-5, 5)
        self.assertEqual((+w).data, tuple(range(-5, 5)))
        self.assertEqual((-w).data, tuple(range(-4, 6))[::-1])
        self.assertEqual(abs(w).data, (5, 4, 3, 2, 1, 0, 1, 2, 3, 4))
        self.assertEqual((~w).data, tuple(range(-5, 5))[::-1])

        w = W()
        self.assertEqual((w & 1).data, (0, 1, 0, 1))
        self.assertEqual((1 & w).data, (0, 1, 0, 1))
        w &= 1
        self.assertEqual(w.data, (0, 1, 0, 1))

        w = W()
        self.assertEqual((w | 1).data, (-1, -1, 1, 1))
        self.assertEqual((1 | w).data, (-1, -1, 1, 1))
        w |= 1
        self.assertEqual(w.data, (-1, -1, 1, 1))

        w = W()
        self.assertEqual((w ^ 1).data, (-1, -2, 1, 0))
        self.assertEqual((1 ^ w).data, (-1, -2, 1, 0))
        w ^= 1
        self.assertEqual(w.data, (-1, -2, 1, 0))

        w = W(0, 4)
        self.assertEqual((w << 1).data, (0, 2, 4, 6))
        self.assertEqual((1 << w).data, (1, 2, 4, 8))
        w <<= 1
        self.assertEqual(w.data, (0, 2, 4, 6))

        w = W(0, 4)
        self.assertEqual((w >> 1).data, (0, 0, 1, 1))
        self.assertEqual((1 >> w).data, (1, 0, 0, 0))
        w >>= 1
        self.assertEqual(w.data, (0, 0, 1, 1))

        w = W()
        self.assertEqual((w + 1).data, (-1, 0, 1, 2))
        self.assertEqual((1 + w).data, (-1, 0, 1, 2))
        w += 1
        self.assertEqual(w.data, (-1, 0, 1, 2))

        w = W()
        self.assertEqual((w - 1).data, (-3, -2, -1, 0))
        self.assertEqual((1 - w).data, (3, 2, 1, 0))
        w -= 1
        self.assertEqual(w.data, (-3, -2, -1, 0))

        w = W()
        self.assertEqual((w * 2).data, (-4, -2, 0, 2))
        self.assertEqual((2 * w).data, (-4, -2, 0, 2))
        w *= 2
        self.assertEqual(w.data, (-4, -2, 0, 2))

        w = ph.Wrapper((-2, -1, 0.2, 1))
        self.assertEqual((w / 2.).data, (-1, -0.5, 0.1, 0.5))
        self.assertEqual((2. / w).data, (-1, -2, 10, 2))
        w /= 2.
        self.assertEqual(w.data, (-1, -0.5, 0.1, 0.5))

        w = ph.Wrapper((-2, -1, 0.2, 1))
        self.assertEqual((w // 2).data, (-1, -1, 0, 0))
        self.assertEqual((2 // w).data, (-1, -2, 9.0, 2))
        w //= 2
        self.assertEqual(w.data, (-1, -1, 0, 0))

        w = W()
        self.assertEqual((w % 2).data, (0, 1, 0, 1))
        w %= 2
        self.assertEqual(w.data, (0, 1, 0, 1))

        w = W()
        self.assertEqual((w**2).data, (4, 1, 0, 1))
        self.assertEqual((2**w).data, (0.25, 0.5, 1, 2))
        w **= 2
        self.assertEqual(w.data, (4, 1, 0, 1))
