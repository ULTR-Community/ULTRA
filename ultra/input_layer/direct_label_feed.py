"""Create batch data directly based on labels.

See the following paper for more information on the simulation data.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json
import numpy as np
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin
from ultra.input_layer import BaseInputFeed
from ultra.utils import click_models as cm
import ultra.utils


class DirectLabelFeed(BaseInputFeed):
    """Feed data with human annotations.

    This class implements a input layer for unbiased learning to rank experiments
    by directly feeding the model with the true labels of each query-document pair.
    """

    def __init__(self, model, batch_size, hparam_str, session=None):
        """Create the model.

        Args:
            model: (BasicModel) The model we are going to train.
            batch_size: the size of the batches generated in each iteration.
            hparam_str: the hyper-parameters for the input layer.
        """
        self.hparams = ultra.utils.hparams.HParams(
            use_max_candidate_num=True,
            # Set to use the maximum number of candidate documents instead of a
            # limited list size.
        )

        self.hparams.parse(hparam_str)

        self.start_index = 0
        self.count = 1
        self.rank_list_size = model.max_candidate_num if self.hparams.use_max_candidate_num else model.rank_list_size
        self.feature_size = model.feature_size
        self.batch_size = batch_size
        self.model = model
        print(
            'Create direct label feed with list size %d with feature size %d' %
            (self.rank_list_size, self.feature_size))

    def prepare_true_labels_with_index(
            self, data_set, index, docid_inputs, letor_features, labels, check_validation=True):
        i = index
        # Generate label list.
        label_list = [
            0 if data_set.initial_list[i][x] < 0 else data_set.labels[i][x] for x in range(
                self.rank_list_size)]

        # Check if data is valid
        if check_validation and sum(label_list) == 0:
            return
        base = len(letor_features)
        for x in range(self.rank_list_size):
            if data_set.initial_list[i][x] >= 0:
                letor_features.append(
                    data_set.features[data_set.initial_list[i][x]])
        docid_inputs.append(list([-1 if data_set.initial_list[i][x]
                                  < 0 else base + x for x in range(self.rank_list_size)]))
        labels.append(label_list)
        return

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

        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
        length = len(data_set.initial_list)
        docid_inputs, letor_features, labels = [], [], []
        rank_list_idxs = []
        for _ in range(self.batch_size):
            i = int(random.random() * length)
            rank_list_idxs.append(i)
            self.prepare_true_labels_with_index(data_set, i,
                                                docid_inputs, letor_features, labels, check_validation)

        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
        # Create info_map to store other information
        info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': docid_inputs,
            'click_list': labels,
            'letor_features': letor_features
        }

        return input_feed, info_map

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
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels = [], [], []

        num_remain_data = len(data_set.initial_list) - index
        for offset in range(min(self.batch_size, num_remain_data)):
            i = index + offset
            self.prepare_true_labels_with_index(
                data_set, i, docid_inputs, letor_features, labels, check_validation)

        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map

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
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels = [], [], []

        i = index
        self.prepare_true_labels_with_index(
            data_set,
            i,
            docid_inputs,
            letor_features,
            labels,
            check_validation)

        letor_features_length = len(letor_features)
        for j in range(self.rank_list_size):
            if docid_inputs[-1][j] < 0:
                docid_inputs[-1][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map
