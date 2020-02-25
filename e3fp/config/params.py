"""Get E3FP default parameters and read parameters from files.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import copy
import ast

try:
    from ConfigParser import (
        SafeConfigParser,
        NoSectionError,
        DuplicateSectionError,
    )
except ImportError:  # Python 3
    from configparser import (
        SafeConfigParser,
        NoSectionError,
        DuplicateSectionError,
    )

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
DEF_PARAM_FILE = os.path.join(CONFIG_DIR, "defaults.cfg")


def read_params(params=None, fill_defaults=False):
    """Get combination of provided parameters and default parameters.

    Parameters
    ----------
    params : str or SafeConfigParser, optional
        User provided parameters as an INI file or `SafeConfigParser`.
        Any parameters provided will replace default parameters.
    fill_defaults : bool, optional
        Fill values that aren't provided with package defaults, if `params`
        is file.

    Returns
    -------
    all_params : SafeConfigParser
        Combination of default and user-provided parameters.
    """
    if isinstance(params, SafeConfigParser):
        return copy.copy(params)

    params_list = []
    if fill_defaults:
        params_list.append(DEF_PARAM_FILE)
    if params is not None:
        params_list.append(params)

    all_params = SafeConfigParser()
    all_params.read(params_list)

    return all_params


def write_params(params, params_file="params.cfg"):
    """Write params to file.

    Parameters
    ----------
    params : SafeConfigParser
        Params
    params_file : str
        Params file
    """
    with open(params_file, "w") as f:
        params.write(f)


def get_value(
    params, section_name, param_name, dtype=str, auto=False, fallback=None
):
    """Get value from params with fallback.

    Parameters
    ----------
    params : SafeConfigParser
        Parameters
    section_name : str
        Name of section in `params`
    param_name : str
        Name of parameter in `section`
    dtype : type, optional
        Type to return data as.
    auto : bool, optional
        Auto-discover type of value. If provided, `dtype` is ignored.
    fallback : any, optional
        Value to return if getting value fails.

    Returns
    -------
    value : any
        Value of parameter or `fallback`.
    """
    if auto:
        try:
            value = params.get(section_name, param_name)
        except ValueError:
            return fallback

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    else:
        get_function = params.get
        if dtype is int:
            get_function = params.getint
        elif dtype is float:
            get_function = params.getfloat
        elif dtype is bool:
            get_function = params.getboolean

        try:
            return get_function(section_name, param_name)
        except ValueError:
            return fallback


def get_default_value(*args, **kwargs):
    global default_params
    return get_value(default_params, *args, **kwargs)


def update_params(
    params_dict, params=None, section_name=None, fill_defaults=False
):
    """Set `SafeConfigParser` values from a sections dict.

    Sections dict key must be parameter sections, and value must be dict
    matching parameter name to value. If existing `SafeConfigParser` is
    provided, parameter values are updated.

    Parameters
    ----------
    params_dict : dict
        If `section_name` is provided, dict must match parameter names to
        values. If `section_name` is not provided, dict key(s) must be
        parameter sections, and value(s) must be parameter dict.
    params : SafeConfigParser, optional
        Existing parameters.
    section_name : str, optional
        Name of section to which to add parameters in `params_dict`
    fill_defaults : bool, optional
        Fill values that aren't provided with package defaults, if `params`
        is file.
    """
    if params is None:
        params = SafeConfigParser()
    else:
        params = read_params(params, fill_defaults=fill_defaults)

    if section_name is not None:
        try:
            params.add_section(section_name)
        except DuplicateSectionError:
            pass

        for param_name, param_value in params_dict.items():
            params.set(section_name, param_name, str(param_value))
    else:
        sections_dict = params_dict
        for section_name, params_dict in sections_dict.items():
            for param_name, param_value in params_dict.items():
                params.set(section_name, param_name, param_value)
    return params


def params_to_sections_dict(params, auto=True):
    """Get dict of sections dicts in params, with optional type discovery.

    Parameters
    ----------
    params : str or SafeConfigParser
        Params to read
    auto : bool, optional
        Auto typing of parameter values.

    Returns
    ----------
    dict : dict matching sections to parameters to values.
    """
    params = read_params(params)
    sections = default_params.sections()
    params_dicts = {}
    for section in sections:
        try:
            params_dict = dict(params.items(section))
        except NoSectionError:
            continue
        if auto:
            params_dict = {
                param_name: get_value(params, section, param_name, auto=True)
                for param_name in params_dict
            }
        params_dicts[section] = params_dict
    return params_dicts


default_params = read_params(fill_defaults=True)
