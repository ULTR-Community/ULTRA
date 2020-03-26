"""The basic class that contains all the API needed for the implementation of a ranking model.
    
"""

from __future__ import print_function
from __future__ import absolute_import
from abc import ABC, abstractmethod
import os,sys
import tensorflow as tf

def selu(x):
    """ Create the scaled exponential linear unit (SELU) activation function. More information can be found in
            Klambauer, G., Unterthiner, T., Mayr, A. and Hochreiter, S., 2017. Self-normalizing neural networks. In Advances in neural information processing systems (pp. 971-980).
        
        Args:
            x: (tf.Tensor) A tensor containing a set of numbers
        
        Returns:
            The tf.Tensor produced by applying SELU on each element in x.
        """
    with tf.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

class ActivationFunctions(object):
  """Activation Functions key strings."""

  ELU = 'elu'

  RELU = 'relu'

  SELU = 'selu'

  TANH = 'tanh'

  SIGMOID = 'sigmoid'

class BaseRankingModel(ABC):

    ACT_FUNC_DIC = {
        ActivationFunctions.ELU: tf.nn.elu,
        ActivationFunctions.RELU: tf.nn.relu,
        ActivationFunctions.SELU: selu,
        ActivationFunctions.TANH: tf.nn.tanh,
        ActivationFunctions.SIGMOID: tf.nn.sigmoid
    }

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

