"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf
import tensorflow_ranking as tfr
from abc import ABC, abstractmethod
from . import ranking_model
from . import metrics
sys.path.append("..")
import utils

class BasicAlgorithm(ABC):
    """The basic class that contains all the API needed for the 
        implementation of an unbiased learning to rank algorithm.

    """

    @abstractmethod
    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.
    
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        self.is_training = None
        self.docid_inputs = None # a list of top documents
        self.letor_features = None # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None # the ranking scores of the inputs
        pass

    @abstractmethod
    def step(self, session, input_feed, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: (tf.Session) tensorflow session to use.
            input_feed: (dictionary) A dictionary containing all the input feed data.
            forward_only: whether to do the backward step (False) or only forward (True).

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        pass
        