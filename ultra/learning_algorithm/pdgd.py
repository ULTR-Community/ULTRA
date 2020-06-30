"""Training and testing the Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Pairwise Differentiable Gradient Descent (PDGD) algorithm.

    * Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.

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

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils as utils
import ultra


class PDGD(BaseAlgorithm):
    """The Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

    This class implements the Pairwise Differentiable Gradient Descent (PDGD) algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise Differentiable Gradient Descent (PDGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate (\mu).
            # Scalar for the probability distribution.
            tau=1,
            max_gradient_norm=1.0,            # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.005,
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
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

        # reshape from [rank_list_size, ?] to [?, rank_list_size]
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
            self.train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            train_labels = self.labels[:self.rank_list_size]
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train_eval'])

            # Build training pair inputs only when it is training
            self.positive_docid_inputs = tf.placeholder(
                tf.int64, shape=[None], name="positive_docid_input")
            self.negative_docid_inputs = tf.placeholder(
                tf.int64, shape=[None], name="negative_docid_input")
            self.pair_weights = tf.placeholder(
                tf.float32, shape=[None], name="pair_weight")
            # Build ranking loss
            pair_scores = self.get_ranking_scores(
                [self.positive_docid_inputs,
                    self.negative_docid_inputs], is_training=self.is_training, scope='ranking_model'
            )

            self.loss = tf.reduce_sum(
                tf.math.multiply(
                    #self.pairwise_cross_entropy_loss(pair_scores[0], pair_scores[1]),
                    tf.reduce_sum(-tf.exp(pair_scores[0]) / (
                        tf.exp(pair_scores[0]) + tf.exp(pair_scores[1])), 1),
                    self.pair_weights
                )
            )
            params = tf.trainable_variables()
            if self.hparams.l2_loss > 0:
                for p in params:
                    self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            # Gradients and SGD update operation for training the model.
            opt = self.optimizer_func(self.hparams.learning_rate)
            self.gradients = tf.gradients(self.loss, params)
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

        self.train_summary = tf.summary.merge_all(key='train')
        self.train_eval_summary = tf.summary.merge_all(key='train_eval')
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

        if not forward_only:
            # Run the model to get ranking scores
            input_feed[self.is_training.name] = False
            rank_outputs = session.run(
                [self.train_output, self.train_eval_summary], input_feed)

            # reduce value to avoid numerical problems
            rank_outputs[0] = np.array(rank_outputs[0])
            rank_outputs[0] = rank_outputs[0] - \
                np.amax(rank_outputs[0], axis=1, keepdims=True)
            exp_ranking_scores = np.exp(self.hparams.tau * rank_outputs[0])

            # Remove scores for padding documents
            letor_features_length = len(input_feed[self.letor_features.name])
            for i in range(len(input_feed[self.labels[0].name])):
                for j in range(self.rank_list_size):
                    # not a valid doc
                    if input_feed[self.docid_inputs[j].name][i] == letor_features_length:
                        exp_ranking_scores[i][j] = 0.0
            # Compute denominator for each position
            denominators = np.cumsum(
                exp_ranking_scores[:, ::-1], axis=1)[:, ::-1]
            sum_log_denominators = np.sum(
                np.log(
                    denominators,
                    out=np.zeros_like(denominators),
                    where=denominators > 0),
                axis=1)
            # Create training pairs based on the ranking scores and the labels
            positive_docids, negative_docids, pair_weights = [], [], []
            for i in range(len(input_feed[self.labels[0].name])):
                # Generate pairs and compute weights
                for j in range(self.rank_list_size):
                    l = self.rank_list_size - 1 - j
                    # not a valid doc
                    if input_feed[self.docid_inputs[l].name][i] == letor_features_length:
                        continue
                    if input_feed[self.labels[l].name][i] > 0:  # a clicked doc
                        for k in range(l + 2):
                            # find a negative/unclicked doc
                            if k < self.rank_list_size and input_feed[self.labels[k]
                                                                      .name][i] < input_feed[self.labels[l].name][i]:
                                # not a valid doc
                                if input_feed[self.docid_inputs[k]
                                              .name][i] == letor_features_length:
                                    continue
                                positive_docids.append(
                                    input_feed[self.docid_inputs[l].name][i])
                                negative_docids.append(
                                    input_feed[self.docid_inputs[k].name][i])
                                flipped_exp_scores = np.copy(
                                    exp_ranking_scores[i])
                                flipped_exp_scores[k] = exp_ranking_scores[i][l]
                                flipped_exp_scores[l] = exp_ranking_scores[i][k]
                                flipped_denominator = np.cumsum(
                                    flipped_exp_scores[::-1])[::-1]

                                sum_log_flipped_denominator = np.sum(
                                    np.log(
                                        flipped_denominator,
                                        out=np.zeros_like(flipped_denominator),
                                        where=flipped_denominator > 0))
                                #p_r = np.prod(rank_prob[i][min_i:max_i+1])
                                #p_rs = np.prod(flipped_rank_prob[min_i:max_i+1])
                                # weight = p_rs / (p_r + p_rs) = 1 / (1 +
                                # (d_rs/d_r)) = 1 / (1 + exp(log_drs - log_dr))
                                weight = 1.0 / \
                                    (1.0 +
                                     np.exp(min(sum_log_flipped_denominator -
                                                sum_log_denominators[i], 20)))
                                if np.isnan(weight):
                                    print('SOMETHING WRONG!!!!!!!')
                                    print(
                                        'sum_log_denominators[i] is nan: ' + str(np.isnan(sum_log_denominators[i])))
                                    print('sum_log_flipped_denominator is nan ' +
                                          str(np.isnan(sum_log_flipped_denominator)))
                                pair_weights.append(weight)
            input_feed[self.positive_docid_inputs.name] = positive_docids
            input_feed[self.negative_docid_inputs.name] = negative_docids
            input_feed[self.pair_weights.name] = pair_weights

            # Train the model
            input_feed[self.is_training.name] = True
            train_outputs = session.run([
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.train_summary  # Summarize statistics.
            ], input_feed)
            summary = utils.merge_TFSummary(
                [rank_outputs[-1], train_outputs[-1]], [0.5, 0.5])

            # loss, no outputs, summary.
            return train_outputs[1], rank_outputs, summary

        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
