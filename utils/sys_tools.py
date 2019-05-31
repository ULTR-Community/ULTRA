import os
import sys
import traceback

def find_class(class_str):
    mod_str, _sep, class_str = class_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))

def create_object(class_str, *args, **kwargs):
    return find_class(class_str)(*args, **kwargs)
