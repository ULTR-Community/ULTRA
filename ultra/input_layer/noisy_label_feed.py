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


class NoisyLabelFeed(BaseInputFeed):
    """Feed data with human annotations.

    This class implements a input layer for unbiased learning to rank experiments
    by feeding the model with noisy labels/features of each query-document pair.
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
            noisy_label_var=-1.0, # Set max variance for noisy labels (<0 means no variance)
            noisy_feature_var=-1.0, # Set max variance for noisy features (<0 means no variance)
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
        
        self.session = None

    def prepare_noisy_labels_with_index(
            self, data_set, index, docid_inputs, letor_features, labels, 
            letor_feature_vars, label_vars, check_validation=True):
        
        i = index
        
        # Generate label list.
        def create_variance_file(data_set, file_path, max_var):
            if not os.path.exists(file_path): 
                # create variance file
                var_lists = []
                for a in range(len(data_set.initial_list)):
                    var_lists.append(np.random.rand(len(data_set.initial_list[a])) * max_var)
                with open(file_path, 'w') as fout:
                    for a in range(len(var_lists)):
                        fout.write(data_set.qids[a] + ' ' + ' '.join([str(b) for b in var_lists[a]]))
                        fout.write('\n')
                return var_lists
            else:
                var_lists = []
                # read variance file
                with open(file_path) as fin:
                    for line in fin:
                        arr = line.strip().split(' ')
                        var_lists.append([float(b) for b in arr[1:]])
                return var_lists

        # check the needs of noisy labels:
        label_list = []
        label_var_list = []
        if self.hparams.noisy_label_var > 0:
            # try to read label_vars from files
            if not hasattr(data_set, 'label_vars'):
                file_path = data_set.data_path + '/label_var_' + str(self.hparams.noisy_label_var) + '.var'
                data_set.label_vars = create_variance_file(data_set, file_path, self.hparams.noisy_label_var)
            # create noisy labels
            label_var_list = [
                0 if data_set.initial_list[i][x] < 0 else data_set.label_vars[i][x] for x in range(
                    self.rank_list_size)]
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
                if self.hparams.noisy_feature_var > 0:
                    # try to read feature_vars from files
                    if not hasattr(data_set, 'feature_vars'):
                        file_path = data_set.data_path + '/feature_var_' + str(self.hparams.noisy_feature_var) + '.var'
                        data_set.feature_vars = create_variance_file(data_set, file_path, self.hparams.noisy_feature_var)
                    # create noisy features
                    letor_feature_vars.append([data_set.feature_vars[i][x] for k in range(len(data_set.features[data_set.initial_list[i][x]]))])
                    
        docid_inputs.append(list([-1 if data_set.initial_list[i][x]
                                  < 0 else base + x for x in range(self.rank_list_size)]))
        labels.append(label_list)
        label_vars.append(label_var_list)
        return

    def add_gaussian_noise(self, means, stds):
        if self.session == None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
        # model
        mean = tf.placeholder(tf.float32, shape=[None, None],
                                                name="mean")  # the letor features for the documents
        std = tf.placeholder(tf.float32, shape=[None, None],
                                                name="std")  # the letor features for the documents
        nor = tf.compat.v1.distributions.Normal(mean, std)
        output = nor.sample()
        # run sampling
        input_feed = {
            mean.name: means,
            std.name: stds,
        }
        output_feed = [output]
        outputs = self.session.run(output_feed, input_feed)
        return outputs[0]
        

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
        docid_inputs, letor_features, labels, letor_feature_vars, label_vars = [], [], [], [], []
        rank_list_idxs = []
        for _ in range(self.batch_size):
            i = int(random.random() * length)
            rank_list_idxs.append(i)
            self.prepare_noisy_labels_with_index(data_set, i,
                                                docid_inputs, letor_features, labels, letor_feature_vars, label_vars, check_validation)

        # Add noise
        if self.hparams.noisy_label_var > 0:
            labels = self.add_gaussian_noise(labels, label_vars)
        if self.hparams.noisy_feature_var > 0:
            letor_features = self.add_gaussian_noise(letor_features, letor_feature_vars)

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

        docid_inputs, letor_features, labels, letor_feature_vars, label_vars = [], [], [], [], []

        num_remain_data = len(data_set.initial_list) - index
        for offset in range(min(self.batch_size, num_remain_data)):
            i = index + offset
            self.prepare_noisy_labels_with_index(
                data_set, i, docid_inputs, letor_features, labels, letor_feature_vars, label_vars, check_validation)

        # Add noise
        if self.hparams.noisy_label_var > 0:
            labels = self.add_gaussian_noise(labels, label_vars)
        if self.hparams.noisy_feature_var > 0:
            letor_features = self.add_gaussian_noise(letor_features, letor_feature_vars)

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

        docid_inputs, letor_features, labels, letor_feature_vars, label_vars = [], [], [], [], []

        i = index
        self.prepare_noisy_labels_with_index(
            data_set,
            i,
            docid_inputs,
            letor_features,
            labels, 
            letor_feature_vars, 
            label_vars,
            check_validation)

        # Add noise
        if self.hparams.noisy_label_var > 0:
            labels = self.add_gaussian_noise(labels, label_vars)
        if self.hparams.noisy_feature_var > 0:
            letor_features = self.add_gaussian_noise(letor_features, letor_feature_vars)

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
