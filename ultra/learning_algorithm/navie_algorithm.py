"""The navie algorithm that directly trains ranking models with clicks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf

from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


class NavieAlgorithm(BaseAlgorithm):
    """The navie algorithm that directly trains ranking models with input labels.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build NavieAlgorithm')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_cross_entropy',            # Select Loss function
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
            # Build model
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            train_labels = self.labels[:self.rank_list_size]

            tf.summary.scalar(
                'Max_output_score',
                tf.reduce_max(train_output),
                collections=['train'])
            tf.summary.scalar(
                'Min_output_score',
                tf.reduce_min(train_output),
                collections=['train'])

            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output)

            tf.summary.scalar(
                'Max_output_score_without_pad',
                tf.reduce_max(pad_removed_train_output),
                collections=['train'])
            tf.summary.scalar(
                'Min_output_score_without_pad',
                tf.reduce_min(pad_removed_train_output),
                collections=['train'])

            self.loss = None
            if self.hparams.loss_func == 'sigmoid_cross_entropy':
                self.loss = self.sigmoid_loss_on_list(
                    train_output, reshaped_train_labels)
            elif self.hparams.loss_func == 'pairwise_loss':
                self.loss = self.pairwise_loss_on_list(
                    train_output, reshaped_train_labels)
            else:
                self.loss = self.softmax_loss(
                    train_output, reshaped_train_labels)
            params = tf.trainable_variables()
            if self.hparams.l2_loss > 0:
                loss_l2 = 0.0
                for p in params:
                    loss_l2 += tf.nn.l2_loss(p)
                tf.summary.scalar(
                    'L2 Loss',
                    tf.reduce_mean(loss_l2),
                    collections=['train'])
                self.loss += self.hparams.l2_loss * loss_l2

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
