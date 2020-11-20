"""Training and testing the Null Space Gradient Descent (NSGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Null Space Gradient Descent (NSGD) algorithm.

    * Huazheng Wang, Ramsey Langley, Sonwoo Kim, Eric McCord-Snook, Hongning Wang. 2018. Efficient Exploration of Gradient Space for Online Learning to Rank. In SIGIR.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
from ultra.learning_algorithm.dbgd import DBGD
import ultra.utils
import ultra


class NSGD(DBGD):
    """The Null Space Gradient Descent (NSGD) algorithm for unbiased learning to rank.

    This class implements the Null Space Gradient Descent (NSGD) algorithm based on the input layer feed. See the following paper for more information on the algorithm.

    * Huazheng Wang, Ramsey Langley, Sonwoo Kim, Eric McCord-Snook, Hongning Wang. 2018. Efficient Exploration of Gradient Space for Online Learning to Rank. In SIGIR.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Null Space Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # The update rate for randomly sampled weights.
            learning_rate=0.5,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            grad_strategy='sgd',        # Select gradient strategy
            # Select number of rankers to try in each batch.
            ranker_num=4,
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)
        self.ranker_num = self.hparams.ranker_num

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.winners = tf.placeholder(tf.float32, shape=[None, self.ranker_num + 1],
                                      name="winners")  # winners of interleaved tests
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)
        self.output = tf.concat(
            self.get_ranking_scores(
                self.docid_inputs,
                is_training=self.is_training,
                scope='ranking_model'),
            1)
        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels))
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                tf.summary.scalar(
                    '%s_%d' %
                    (metric, topn), metric_value, collections=['eval'])

        # Build model
        if not forward_only:
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            train_output = tf.concat(
                self.get_ranking_scores(
                    self.docid_inputs[:self.rank_list_size],
                    is_training=self.is_training,
                    scope='ranking_model'),
                1)
            train_labels = self.labels[:self.rank_list_size]

            ranking_model_params = self.model.model_parameters
            # Create memory to store historical bad noisy parameters
            bad_noisy_params = {}
            for x in ranking_model_params:
                if x not in bad_noisy_params:
                    bad_noisy_params[x] = []
                    for i in range(self.ranker_num):
                        bad_noisy_params[x].append(tf.Variable(
                            tf.zeros(ranking_model_params[x].get_shape()), trainable=False))

            # Compute null space
            def compute_null_space(param_list):
                original_shape = param_list[0].get_shape()
                flatten_list = [tf.reshape(param_list[i], [1, -1])
                                for i in range(len(param_list))]
                matrix = tf.stack(flatten_list, axis=1)
                # print(matrix.get_shape())
                s, u, v = tf.linalg.svd(matrix)
                # find null space vector
                mask = tf.cast(tf.math.equal(s, 0), dtype=tf.float32)
                # print(mask.get_shape())
                # param_num x ranker_num
                return (tf.squeeze(v * mask), original_shape)

            null_space_dict = {}
            for x in bad_noisy_params:
                null_space_dict[x] = compute_null_space(bad_noisy_params[x])

            def sample_from_null_space(null_space_matrix, original_shape):
                if sum([original_shape[i].value for i in range(
                        original_shape.rank)]) > 1:
                    sampled_vector = tf.reduce_sum(
                        null_space_matrix * tf.random.normal([1, self.ranker_num]), axis=1)
                    return tf.math.l2_normalize(
                        tf.reshape(sampled_vector, original_shape))
                else:
                    return tf.math.l2_normalize(
                        tf.random.normal(original_shape))

            new_output_lists = []
            params = []
            param_gradient_from_rankers = {}
            # noise_lists = [tf.zeros_like(self.output, tf.float32)]
            for i in range(self.ranker_num):
                # Create unit noise from null space
                noisy_params = {}
                for x in ranking_model_params:
                    noisy_params[x] = sample_from_null_space(
                        null_space_dict[x][0], null_space_dict[x][1])

                # Apply the noise to get new ranking scores
                new_output_list = None
                if self.hparams.need_interleave:  # compute scores on whole list if needs interleave
                    new_output_list = self.get_ranking_scores(
                        self.docid_inputs, is_training=self.is_training, scope='ranking_model', noisy_params=noisy_params, noise_rate=self.hparams.learning_rate)
                else:
                    new_output_list = self.get_ranking_scores(
                        self.docid_inputs[:self.rank_list_size], is_training=self.is_training, scope='ranking_model', noisy_params=noisy_params, noise_rate=self.hparams.learning_rate)
                new_output_lists.append(tf.concat(new_output_list, 1))
                for x in noisy_params:
                    if x not in param_gradient_from_rankers:
                        params.append(ranking_model_params[x])
                        param_gradient_from_rankers[x] = [
                            tf.zeros_like(ranking_model_params[x])]
                    param_gradient_from_rankers[x].append(noisy_params[x])

            # Compute NDCG for the old ranking scores.
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            previous_ndcg = ultra.utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(
                reshaped_train_labels, train_output, None)
            self.loss = 1.0 - previous_ndcg

            final_winners = None
            if self.hparams.need_interleave:  # Use result interleaving
                self.output = [self.output] + new_output_lists
                final_winners = self.winners
            else:  # No result interleaving
                score_lists = [train_output] + new_output_lists
                ndcg_lists = []
                for scores in score_lists:
                    ndcg = ultra.utils.make_ranking_metric_fn(
                        'ndcg', self.rank_list_size)(
                        reshaped_train_labels, scores, None)
                    ndcg_lists.append(ndcg - previous_ndcg)
                ndcg_gains = tf.ceil(tf.stack(ndcg_lists))
                final_winners = ndcg_gains / \
                    (tf.reduce_sum(ndcg_gains, axis=0) + 0.000000001)

            # Compute gradients
            self.gradients = []
            self.update_ops_list = []
            for p in params:
                gradient_matrix = tf.expand_dims(
                    tf.stack(param_gradient_from_rankers[p.name]), axis=0)
                expended_winners = final_winners
                for i in range(gradient_matrix.get_shape().rank -
                               expended_winners.get_shape().rank):
                    expended_winners = tf.expand_dims(
                        expended_winners, axis=-1)
                # Sum up to get the final gradient
                self.gradients.append(
                    tf.reduce_mean(
                        tf.reduce_sum(
                            expended_winners * gradient_matrix,
                            axis=1
                        ),
                        axis=0)
                )
                # Update historical bad nosiy parameters
                expended_losers = tf.cast(
                    tf.math.equal(
                        tf.reduce_sum(
                            expended_winners,
                            axis=0,
                            keepdims=True),
                        0),
                    dtype=tf.float32)
                bad_noise_list = tf.unstack(
                    tf.squeeze(
                        expended_losers *
                        gradient_matrix),
                    axis=0)[
                    1:]
                for i in range(self.ranker_num):
                    bad_noise_list[i] = tf.reshape(
                        bad_noise_list[i], bad_noisy_params[p.name][i].get_shape())
                    self.update_ops_list.append(
                        bad_noisy_params[p.name][i].assign(bad_noise_list[i])
                    )

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            # Gradients and SGD update operation for training the model.
            opt = self.optimizer_func(self.hparams.learning_rate)
            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                           self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                                   global_step=self.global_step)
                tf.summary.scalar(
                    'Gradient Norm',
                    self.norm,
                    collections=['train'])
            else:
                self.norm = None
                self.updates = opt.apply_gradients(zip(self.gradients, params),
                                                   global_step=self.global_step)

            tf.summary.scalar(
                'Learning Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar('Loss', self.loss, collections=['train'])
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

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
        #print ("!!!!!!!!!!!!!", tf.shape(self.new_output))

        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.train_summary  # Summarize statistics.
            ] + self.update_ops_list
            outputs = session.run(output_feed, input_feed)
            # loss, no outputs, summary.
            return outputs[1], None, outputs[2]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
