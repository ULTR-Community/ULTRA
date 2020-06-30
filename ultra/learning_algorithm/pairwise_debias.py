"""Training and testing the Pairwise Debiasing algorithm for unbiased learning to rank.

See the following paper for more information on the Pairwise Debiasing algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

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
import ultra.utils


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return tf.ceil(probs - tf.random_uniform(tf.shape(probs)))


class PairDebias(BaseAlgorithm):
    """The Pairwise Debiasing algorithm for unbiased learning to rank.

    This class implements the Pairwise Debiasing algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise Debiasing algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            EM_step_size=0.05,                  # Step size for EM algorithm.
            learning_rate=0.005,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            # An int specify the regularization term.
            regulation_p=1,
            # Set strength for L2 regularization.
            l2_loss=0.0,
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

        self.output = self.ranking_model(
            self.max_candidate_num, scope='ranking_model')
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

        # Build unbiased pairwise loss only when it is training
        if not forward_only:
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            train_labels = self.labels[:self.rank_list_size]
            # Build propensity parameters
            self.t_plus = tf.Variable(
                tf.ones([1, self.rank_list_size]), trainable=False)
            self.t_minus = tf.Variable(
                tf.ones([1, self.rank_list_size]), trainable=False)
            self.splitted_t_plus = tf.split(
                self.t_plus, self.rank_list_size, axis=1)
            self.splitted_t_minus = tf.split(
                self.t_minus, self.rank_list_size, axis=1)
            for i in range(self.rank_list_size):
                tf.summary.scalar(
                    't_plus Probability %d' %
                    i,
                    tf.reduce_max(
                        self.splitted_t_plus[i]),
                    collections=['train'])
                tf.summary.scalar(
                    't_minus Probability %d' %
                    i,
                    tf.reduce_max(
                        self.splitted_t_minus[i]),
                    collections=['train'])

            # Build pairwise loss based on clicks (0 for unclick, 1 for click)
            output_list = tf.split(train_output, self.rank_list_size, axis=1)
            t_plus_loss_list = [0.0 for _ in range(self.rank_list_size)]
            t_minus_loss_list = [0.0 for _ in range(self.rank_list_size)]
            self.loss = 0.0
            for i in range(self.rank_list_size):
                for j in range(self.rank_list_size):
                    if i == j:
                        continue
                    valid_pair_mask = tf.math.minimum(
                        tf.ones_like(
                            self.labels[i]), tf.nn.relu(
                            self.labels[i] - self.labels[j]))
                    pair_loss = tf.reduce_sum(
                        valid_pair_mask *
                        self.pairwise_cross_entropy_loss(
                            output_list[i], output_list[j])
                    )
                    t_plus_loss_list[i] += pair_loss / self.splitted_t_minus[j]
                    t_minus_loss_list[j] += pair_loss / self.splitted_t_plus[i]
                    self.loss += pair_loss / \
                        self.splitted_t_plus[i] / self.splitted_t_minus[j]

            # Update propensity
            self.update_propensity_op = tf.group(
                self.t_plus.assign(
                    (1 - self.hparams.EM_step_size) * self.t_plus + self.hparams.EM_step_size * tf.pow(
                        tf.concat(t_plus_loss_list, axis=1) / t_plus_loss_list[0], 1 / (self.hparams.regulation_p + 1))
                ),
                self.t_minus.assign(
                    (1 - self.hparams.EM_step_size) * self.t_minus + self.hparams.EM_step_size * tf.pow(tf.concat(
                        t_minus_loss_list, axis=1) / t_minus_loss_list[0], 1 / (self.hparams.regulation_p + 1))
                )
            )

            # Add l2 loss
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
            tf.summary.scalar(
                'Loss', tf.reduce_mean(
                    self.loss), collections=['train'])

            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
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

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.update_propensity_op,
                self.train_summary  # Summarize statistics.
            ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
