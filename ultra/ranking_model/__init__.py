# note:
from __future__ import absolute_import
from .base_ranking_model import *

from .DNN import *
from .Linear import *
from .DLCM import *
from .GSF import *
from .SetRank import *

def list_available() -> list:
    from .base_ranking_model import BaseRankingModel
    from ultra.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseRankingModel)
