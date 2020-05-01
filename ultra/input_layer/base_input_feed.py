"""The basic class that contains all the API needed for the implementation of a input data feed.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import math
import os
import random
import sys
import time
import json
import numpy as np
import ultra.utils.click_models as cm
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin


class BaseInputFeed(ABC):
    """

    This class implements a input layer for unbiased learning to rank experiments.
    """
    MAX_SAMPLE_ROUND_NUM = 100

    @staticmethod
    def preprocess_data(data_set, hparam_str, exp_settings):
        """Preprocess the data for model creation based on the input feed.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            hparam_str: the hyper-parameters for the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        return

    @abstractmethod
    def __init__(self, model, batch_size, hparam_str, session):
        """Create the model.

        Args:
            model: (BasicModel) The model we are going to train.
            batch_size: the size of the batches generated in each iteration.
            hparam_str: the hyper-parameters for the input layer.
            session: the current tensorflow Session (used for online learning).
        """
        pass

    @abstractmethod
    def get_batch(self, data_set, check_validation=False):
        """Get a random batch of data, prepare for step. Typically used for training.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        pass

    @abstractmethod
    def get_next_batch(self, index, data_set, check_validation=False):
        """Get the next batch of data from a specific index, prepare for step.
           Typically used for validation.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            index: the index of the data before which we will use to create the data batch.
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        pass

    @abstractmethod
    def get_data_by_index(self, data_set, index, check_validation=False):
        """Get one data from the specified index, prepare for step.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            index: the index of the data
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            The triple (docid_inputs, decoder_inputs, target_weights) for
            the constructed batch that has the proper format to call step(...) later.
        """
        pass
