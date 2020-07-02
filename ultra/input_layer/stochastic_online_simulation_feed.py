"""Simulate online learning process and click data based on human annotations.

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
from ultra.input_layer import BaseInputFeed
from ultra.utils import click_models as cm
from ultra.utils.team_draft_interleave import TeamDraftInterleaving
import ultra

import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin


class StochasticOnlineSimulationFeed(BaseInputFeed):
    """Simulate online learning to rank and click data based on human annotations.

    This class implements a input layer for online learning to rank experiments
    by simulating click data based on both the human relevance annotation of
    each query-document pair and a predefined click model.
    """

    def __init__(self, model, batch_size, hparam_str, session):
        """Create the model.

        Args:
            model: (BasicModel) The model we are going to train.
            batch_size: the size of the batches generated in each iteration.
            hparam_str: the hyper-parameters for the input layer.
            session: the current tensorflow Session (used for online learning).
        """
        self.hparams = ultra.utils.hparams.HParams(
            # the setting file for the predefined click models.
            click_model_json='./example/ClickModel/pbm_0.1_1.0_4_1.0.json',
            # Scalar for the probability distribution.
            tau=1,
            # Set True to feed relevance labels instead of simulated clicks.
            oracle_mode=False,
            # Set eta change step for dynamic bias severity in training, 0.0
            # means no change.
            dynamic_bias_eta_change=0.0,
            # Set how many steps to change eta for dynamic bias severity in
            # training, 0.0 means no change.
            dynamic_bias_step_interval=1000,
        )

        print('Create online stochastic simluation feed')
        print(hparam_str)
        self.hparams.parse(hparam_str)
        self.click_model = None
        with open(self.hparams.click_model_json) as fin:
            model_desc = json.load(fin)
            self.click_model = cm.loadModelFromJson(model_desc)
        self.start_index = 0
        self.count = 1
        self.rank_list_size = model.rank_list_size
        self.max_candidate_num = model.max_candidate_num
        self.feature_size = model.feature_size
        self.batch_size = batch_size
        self.model = model
        self.session = session
        self.global_batch_count = 0

        # Check whether the model needs result interleaving.
        self.need_interleave = False
        if hasattr(model.hparams, 'need_interleave'):
            self.need_interleave = model.hparams.need_interleave
            print('Online simulation with interleaving: %s' %
                  (str(self.need_interleave)))
        if self.need_interleave:
            self.interleaving = TeamDraftInterleaving()

    def prepare_true_labels_with_index(
            self, data_set, index, docid_inputs, letor_features, labels, check_validation=False):
        i = index
        # Generate label list.
        label_list = [
            0 if data_set.initial_list[i][x] < 0 else data_set.labels[i][x] for x in range(
                self.max_candidate_num)]

        # Check if data is valid
        if check_validation and sum(label_list) == 0:
            return
        base = len(letor_features)
        for x in range(self.max_candidate_num):
            if data_set.initial_list[i][x] >= 0:
                letor_features.append(
                    data_set.features[data_set.initial_list[i][x]])
        docid_inputs.append(list([-1 if data_set.initial_list[i][x]
                                  < 0 else base + x for x in range(self.max_candidate_num)]))
        labels.append(label_list)

    def simulate_clicks_online(self, input_feed, check_validation=False):
        """Simulate online environment by reranking documents and collect clicks.

        Args:
            input_feed: (dict) The input_feed data.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        # Compute ranking scores with input_feed
        input_feed[self.model.is_training.name] = False
        rank_scores = self.session.run([self.model.output], input_feed)[0]
        # Rerank documents and collect clicks
        letor_features_length = len(input_feed[self.model.letor_features.name])
        local_batch_size = len(input_feed[self.model.docid_inputs[0].name])

        if self.need_interleave:
            input_feed[self.model.winners.name] = [
                None for _ in range(local_batch_size)]

        for i in range(local_batch_size):
            # Get valid doc index
            valid_idx = self.max_candidate_num - 1
            while valid_idx > -1:
                if input_feed[self.model.docid_inputs[valid_idx]
                              .name][i] < letor_features_length:  # a valid doc
                    break
                valid_idx -= 1
            list_len = valid_idx + 1

            def plackett_luce_sampling(score_list):
                # Sample document ranking
                scores = np.array(score_list[:list_len])
                scores = scores - max(scores)
                exp_scores = np.exp(self.hparams.tau * scores)
                probs = exp_scores / np.sum(exp_scores)
                re_list = np.random.choice(np.arange(list_len),
                                           replace=False,
                                           p=probs,
                                           size=np.count_nonzero(probs))
                # Append unselected documents to the end
                used_indexs = set(re_list)
                unused_indexs = []
                for tmp_index in range(list_len):
                    if tmp_index not in used_indexs:
                        unused_indexs.append(tmp_index)
                re_list = np.append(re_list, unused_indexs).astype(int)
                return re_list

            rerank_list = None
            if self.need_interleave:
                # Rerank documents via interleaving
                rank_lists = []
                for j in range(len(rank_scores)):
                    scores = rank_scores[j][i][:list_len]
                    rank_list = plackett_luce_sampling(scores)
                    rank_lists.append(rank_list)

                rerank_list = self.interleaving.interleave(
                    np.asarray(rank_lists))
            else:
                rerank_list = plackett_luce_sampling(rank_scores[i])

            # Rerank documents
            new_docid_list = np.zeros(list_len)
            new_label_list = np.zeros(list_len)
            for j in range(list_len):
                new_docid_list[j] = input_feed[self.model.docid_inputs[rerank_list[j]].name][i]
                new_label_list[j] = input_feed[self.model.labels[rerank_list[j]].name][i]
            # Collect clicks online
            click_list = None
            if self.hparams.oracle_mode:
                click_list = new_label_list[:self.rank_list_size]
            else:
                click_list, _, _ = self.click_model.sampleClicksForOneList(
                    new_label_list[:self.rank_list_size])
                sample_count = 0
                while check_validation and sum(
                        click_list) == 0 and sample_count < self.MAX_SAMPLE_ROUND_NUM:
                    click_list, _, _ = self.click_model.sampleClicksForOneList(
                        new_label_list[:self.rank_list_size])
                    sample_count += 1
            # update input_feed
            for j in range(list_len):
                input_feed[self.model.docid_inputs[j].name][i] = new_docid_list[j]
                if j < self.rank_list_size:
                    input_feed[self.model.labels[j].name][i] = click_list[j]
                else:
                    input_feed[self.model.labels[j].name][i] = 0

            if self.need_interleave:
                # Infer winner in interleaving
                input_feed[self.model.winners.name][i] = self.interleaving.infer_winner(
                    click_list)

        return input_feed

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
            for j in range(self.max_candidate_num):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
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
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

        # Create info_map to store other information
        info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': docid_inputs,
            'click_list': labels,
            'letor_features': letor_features
        }

        self.global_batch_count += 1
        if self.hparams.dynamic_bias_eta_change != 0:
            if self.global_batch_count % self.hparams.dynamic_bias_step_interval == 0:
                self.click_model.eta += self.hparams.dynamic_bias_eta_change
                self.click_model.setExamProb(self.click_model.eta)
                print(
                    'Dynamically change bias severity eta to %.3f' %
                    self.click_model.eta)

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
            for j in range(self.max_candidate_num):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
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
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

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
        for j in range(self.max_candidate_num):
            if docid_inputs[-1][j] < 0:
                docid_inputs[-1][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
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
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map
