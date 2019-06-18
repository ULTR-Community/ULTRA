"""The basic class that contains all the API needed for the implementation of a ranking model.
    
"""

from __future__ import print_function
from __future__ import absolute_import
from abc import ABC, abstractmethod
import os,sys
import tensorflow as tf

class BasicRankingModel(ABC):

    @abstractmethod
    def __init__(self, hparams_str):
        """Create the network.
    
        Args:
            hparams_str: (string) The hyper-parameters used to build the network.
        """
        pass

    @abstractmethod
    def build(self, input_list):
        """ Create the model
        
        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features 
                        for a list of documents.
            reuse: (bool) A flag indicating whether we need to reuse the parameters.
        
        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        pass