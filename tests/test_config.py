"""Tests for loading config files.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os


class TestConfig:
    def test_config_file_exists(self):
        from e3fp.config.params import DEF_PARAM_FILE

        assert os.path.isfile(DEF_PARAM_FILE)
