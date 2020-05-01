import os
import sys
import traceback
import inspect


def find_class(class_str):
    """Find the corresponding class based on a string of class name.

      Args:
        class_str: a string containing the name of the class
      Raises:
        ValueError: If there is no class with the name.
    """
    mod_str, _sep, class_str = class_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def create_object(class_str, *args, **kwargs):
    """Find the corresponding class based on a string of class name and create an object.

      Args:
        class_str: a string containing the name of the class
      Raises:
        ValueError: If there is no class with the name.
    """
    return find_class(class_str)(*args, **kwargs)


def list_recursive_concrete_subclasses(base):
    """List all concrete subclasses of `base` recursively.

      Args:
        base: a string containing the name of the class
    """

    return _filter_concrete(_bfs(base))


def _filter_concrete(classes):
    return list(filter(lambda c: not inspect.isabstract(c), classes))


def _bfs(base):
    return base.__subclasses__() + sum([
        _bfs(subclass)
        for subclass in base.__subclasses__()
    ], [])
