"""Get E3FP default parameters.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import copy
try:
    from ConfigParser import SafeConfigParser
except ImportError:  # Python 3
    from configparser import SafeConfigParser

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
DEF_PARAM_FILE = os.path.join(CONFIG_DIR, "defaults.cfg")


def default_params():
    """Get package default parameters.

    Returns
    -------
    default_params : SafeConfigParser
        Default parameters
    """
    return read_params()


def read_params(params=None):
    """Get combination of provided parameters and default parameters.

    Parameters
    ----------
    params : str or SafeConfigParser, optional
        User provided parameters as an INI file or ``SafeConfigParser``.
        Any parameters provided will replace default parameters.

    Returns
    -------
    all_params : SafeConfigParser
        Combination of default and user-provided parameters.
    """
    if isinstance(params, SafeConfigParser):
        return copy(params)

    all_params = SafeConfigParser()
    if params is None:
        all_params.read(DEF_PARAM_FILE)
    else:
        all_params.read([DEF_PARAM_FILE, params])

    return all_params


def get_value(params, section_name, param_name, dtype=str, fallback=None):
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
    fallback : any, optional
        Value to return if getting value fails.

    Returns
    -------
    value : any
        Value of parameter or `fallback`.
    """
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
    params = default_params()
    return get_value(params, *args, **kwargs)


def update_params(params_dict, params=None, section_name=None):
    """Set ``SafeConfigParser`` values from a sections dict.

    Sections dict key must be parameter sections, and value must be dict
    matching parameter name to value. If existing ``SafeConfigParser`` is
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
    """
    if params is None:
        params = default_params()
    else:
        params = read_params(params)

    if section_name is not None:
        for param_name, param_value in params_dict.iterkeys():
            params.set(section_name, param_name, param_value)
    else:
        sections_dict = params_dict
        for section_name, params_dict in sections_dict.iteritems():
            for param_name, param_value in params_dict.iterkeys():
                params.set(section_name, param_name, param_value)
    return params
