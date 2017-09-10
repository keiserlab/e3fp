"""Utility classes/methods.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import inspect
import warnings


class E3FPWarning(Warning):

    """Base E3FP warning class.

    Unlike normal warnings, these are by default always set to on.
    """


# Always show custom warnings for this package
warnings.filterwarnings('always', category=E3FPWarning)


class E3FPDeprecationWarning(E3FPWarning, DeprecationWarning):

    """A warning class for a deprecated method or class."""


class deprecated(object):

    """Decorator to mark a function as deprecated.

    Issue a deprecation warning when a function is called, and update the
    documentation. A deprecation version must be provided.

    Usage
    -----
    >>> @deprecated("1.1", remove_version="1.3", msg="Function no longer needed")
    ... def my_function():
    ...     pass

    Note
    ----
    Adapted from https://wiki.python.org/moin/PythonDecoratorLibrary
    """

    def __init__(self, deprecated_version, remove_version=None, msg=None):
        """Constructor.

        Parameters
        ----------
        deprecated_version : str
            Version in which object was deprecated (e.g. '1.1')
        remove_version : str, optional
            Version in which object will be removed (e.g. '1.2'). If not
            specified, it is assumed the object will be removed in the next
            release (e.g. '1.2' if `deprecated_version` is '1.1')
        msg : str, optional
            Message to include with deprecation warning, to explain deprecation
            or point to newer version.
        """
        self.deprecated_version = deprecated_version
        if remove_version is None:
            version_info = deprecated_version.split('.')
            version_info[1] = str(int(version_info[1]) + 1)
            for i in range(2, len(version_info)):
                version_info[i] = "0"
            remove_version = ".".join(version_info)
        self.remove_version = remove_version
        if msg is None:
            self.extra = ""
        else:
            self.extra = " {0}".format(msg)

    def __call__(self, obj):
        if inspect.isfunction(obj):
            return self.deprecate_function(obj)
        else:
            raise ValueError("Deprecated object is not a function.")

    def deprecate_function(self, f):
        """Return the decorated function."""
        msg = ("Function `{0}` was deprecated in {1} and will be removed "
               "in {2}. Please update your scripts.{3}").format(
                   f.__name__, self.deprecated_version, self.remove_version,
                   self.extra)

        def new_func(*args, **kwargs):
            warnings.warn(msg, category=E3FPDeprecationWarning, stacklevel=2)
            return f(*args, **kwargs)

        new_func.__name__ = f.__name__
        new_func.__dict__ = f.__dict__
        new_func.__doc__ = f.__doc__
        self.update_docstring(new_func)
        return new_func

    def update_docstring(self, obj):
        """Add deprecation note to docstring."""
        msg = (".. note:: Deprecated in e3fp {0}\n"
               "          `{1}` will be removed in e3fp {2}.{3}").format(
                   self.deprecated_version, obj.__name__, self.remove_version,
                   self.extra)
        obj.__doc__ = "{0}\n\n{1}".format(obj.__doc__, msg)
        return obj
