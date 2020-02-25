"""Tests for util methods.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import unittest
import warnings


class UtilTestCases(unittest.TestCase):
    def test_deprecated(self):
        from e3fp.util import deprecated, E3FPDeprecationWarning

        @deprecated("1.1", remove_version="1.3", msg="DEPRECATED!!!")
        def dep_method():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dep_method()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, E3FPDeprecationWarning))
            message = str(w[-1].message)
            self.assertTrue("deprecated in 1.1" in message)
            self.assertTrue("removed in 1.3" in message)
            self.assertTrue("DEPRECATED!!!" in str(w[-1].message))

        self.assertTrue(
            ".. note:: Deprecated in e3fp 1.1" in dep_method.__doc__
        )
        self.assertTrue("will be removed in e3fp 1.3" in dep_method.__doc__)
        self.assertTrue("DEPRECATED!!!" in dep_method.__doc__)

    def test_efficiency_warning(self):
        from e3fp.util import E3FPEfficiencyWarning

        def test(warn=False):
            if warn:
                raise E3FPEfficiencyWarning("Inefficient!")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            test(warn=False)

            with self.assertRaises(E3FPEfficiencyWarning):
                test(warn=True)


if __name__ == "__main__":
    unittest.main()
