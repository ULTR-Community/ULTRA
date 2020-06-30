"""Training and testing the regression-based EM algorithm for unbiased learning to rank.

See the following paper for more information on the regression-based EM algorithm.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

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


class RegressionEM(BaseAlgorithm):
    """The regression-based EM algorithm for unbiased learning to rank.

    This class implements the regression-based EM algorithm based on the input layer
    feed. See the following paper for more information.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

    In particular, we use the online EM algorithm for the parameter estimations:

    * Cappé, Olivier, and Eric Moulines. "Online expectation–maximization algorithm for latent data models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Regression-based EM algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            EM_step_size=0.05,                  # Step size for EM algorithm.
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
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

        # Build model
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

        if not forward_only:
            # Build EM graph only when it is training
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            sigmoid_prob_b = tf.Variable(tf.ones([1]) - 1.0)
            train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            train_output = train_output + sigmoid_prob_b
            train_labels = self.labels[:self.rank_list_size]
            self.propensity = tf.Variable(
                tf.ones([1, self.rank_list_size]) * 0.9, trainable=False)

            self.splitted_propensity = tf.split(
                self.propensity, self.rank_list_size, axis=1)
            for i in range(self.rank_list_size):
                tf.summary.scalar(
                    'Examination Probability %d' %
                    i,
                    tf.reduce_max(
                        self.splitted_propensity[i]),
                    collections=['train'])

            # Conduct estimation step.
            gamma = tf.sigmoid(train_output)
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            p_e1_r1_c1 = 1
            p_e1_r0_c0 = self.propensity * \
                (1 - gamma) / (1 - self.propensity * gamma)
            p_e0_r1_c0 = (1 - self.propensity) * gamma / \
                (1 - self.propensity * gamma)
            p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / \
                (1 - self.propensity * gamma)
            p_e1 = p_e1_r0_c0 + p_e1_r1_c1
            p_r1 = reshaped_train_labels + \
                (1 - reshaped_train_labels) * p_e0_r1_c0

            # Conduct maximization step
            self.update_propensity_op = self.propensity.assign(
                (1 - self.hparams.EM_step_size) * self.propensity + self.hparams.EM_step_size * tf.reduce_mean(
                    reshaped_train_labels + (1 - reshaped_train_labels) * p_e1_r0_c0, axis=0, keep_dims=True
                )
            )

            # Get Bernoulli samples and compute rank loss
            self.ranker_labels = get_bernoulli_sample(p_r1)
            self.loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.ranker_labels, logits=train_output),
                    axis=1
                )
            )
            # record additional positive instance from sampling
            split_ranker_labels = tf.split(
                self.ranker_labels, self.rank_list_size, axis=1)
            for i in range(self.rank_list_size):
                additional_postive_instance = (tf.reduce_sum(split_ranker_labels[i]) - tf.reduce_sum(
                    train_labels[i])) / (tf.reduce_sum(tf.ones_like(train_labels[i])) - tf.reduce_sum(train_labels[i]))
                tf.summary.scalar(
                    'Additional pseudo clicks %d' %
                    i, additional_postive_instance, collections=['train'])

            self.propensity_weights = 1.0 / self.propensity
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
            clipped_labels = tf.clip_by_value(
                reshaped_train_labels, clip_value_min=0, clip_value_max=1)
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    list_weights = tf.reduce_mean(
                        self.propensity_weights * clipped_labels, axis=1, keep_dims=True)
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train'])
                    weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, list_weights)
                    tf.summary.scalar(
                        'Weighted_%s_%d' %
                        (metric, topn), weighted_metric_value, collections=['train'])

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
