# -*- coding: utf-8 -*-


__all__ = ["UtilTest"]


import unittest

import numphy as ph


class UtilTest(unittest.TestCase):

    def test_empty_slice(self):
        self.assertEqual(ph.util.empty_slice, slice(None))

    def test_ellipsis_expansion(self):
        seq = (1, 2, 3, Ellipsis, 6)

        self.assertEqual(ph.util.expand_ellipsis(seq, 5), (1, 2, 3, slice(None), 6))
        self.assertEqual(ph.util.expand_ellipsis(seq, 6), (1, 2, 3, slice(None), slice(None), 6))

        with self.assertRaises(Exception):
            ph.util.expand_ellipsis(seq, 4)

    def test_no_value(self):
        self.assertIsInstance(ph.util.no_value, ph.util.NoValue)
        self.assertFalse(ph.util.no_value)
