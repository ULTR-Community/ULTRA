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
    def build(self, input_list, is_training=False):
        """ Create the model
        
        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features 
                        for a list of documents.
            is_training: (bool) A flag indicating whether the model is running in training mode.
        
        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        pass
    
    @abstractmethod
    def build_with_random_noise(self, input_list, noise_rate, is_training):
        """ Create the model and add random noise (for online learning).
        
        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features 
                        for a list of documents.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.
        
        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
            A list of (tf.Tensor, tf.Tensor) containing the random noise and the parameters it is designed for.
        """
        pass