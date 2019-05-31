from __future__ import print_function
from __future__ import absolute_import
import os,sys
import tensorflow as tf

class BasicRankingModel:
    def __init__(self, hparams_str):
        """Create the network.
    
        Args:
            hparams_str: (string) The hyper-parameters used to build the network.
        """

        self.hparams = tf.contrib.training.HParams(
        )
        self.hparams.parse(hparams_str)

    def build(self, input_list):
        """ Create the model
        
        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features 
                        for a list of documents.
            reuse: (bool) A flag indicating whether we need to reuse the parameters.
        """
        return None