"""Tests for loading config files.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import unittest


class ConfigTestCases(unittest.TestCase):
    def test_config_file_exists(self):
        from e3fp.config.params import DEF_PARAM_FILE

        self.assertTrue(os.path.isfile(DEF_PARAM_FILE))


if __name__ == "__main__":
    unittest.main()
