"""Training and testing the Multileave Gradient Descent (MGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Multileave Gradient Descent (MGD) algorithm.

    * Anne Schuth, Harrie Oosterhuis, Shimon Whiteson, Maarten de Rijke. 2016. Multileave Gradient Descent for Fast Online Learning to Rank. In WSDM. 457-466.

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
import tensorflow_ranking as tfr
import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
from ultra.learning_algorithm.dbgd_interleave import DBGDInterleave
import ultra.utils
import ultra


class MGDInterleave(DBGDInterleave):
    """The Multileave Gradient Descent (MGD) algorithm for unbiased learning to rank.

    This class implements the Multileave Gradient Descent (MGD) algorithm based on the input layer feed. See the following paper for more information on the algorithm.

    * Anne Schuth, Harrie Oosterhuis, Shimon Whiteson, Maarten de Rijke. 2016. Multileave Gradient Descent for Fast Online Learning to Rank. In WSDM. 457-466.

    """
    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Dueling Bandit Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # The update rate for randomly sampled weights.
            noise_rate=0.5,
            learning_rate=0.01,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            # Set strength for L2 regularization.
            l2_loss=0.01,
            grad_strategy='ada',            # Select gradient strategy
            ranker_num=5,
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
        self.winners = tf.placeholder(tf.int32, shape=[None],
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
            self.rank_list_size = exp_settings['train_list_cutoff']
            train_output = tf.concat(
                self.get_ranking_scores(
                    self.docid_inputs,
                    is_training=self.is_training,
                    scope='ranking_model'),
                1)
            train_labels = self.labels[:self.rank_list_size]
            # Create random gradients and apply it to get new ranking scores
            # new_output_lists = [self.output]
            new_output_lists = []
            param_lists = []
            noise_lists = []
            # noise_lists = [tf.zeros_like(self.output, tf.float32)]
            for i in range(self.ranker_num):
                new_output_list, noise_list = self.get_ranking_scores_with_noise(
                    self.docid_inputs, is_training=self.is_training, scope='ranking_model')
                new_output_lists.append(new_output_list)
                param = [p[1] for p in noise_list]
                noise = [p[0] for p in noise_list]
                noise_lists.append(noise)
                param_lists.append(param)
            
            
            # Compute NDCG for the old ranking scores and new ranking scores
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            self.new_output = tf.concat(tf.convert_to_tensor(new_output_lists), 2)
            print (tf.shape(reshaped_train_labels), tf.shape(self.new_output))

            self.output = (self.output, train_output, self.new_output)

            previous_ndcg = ultra.utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(
                reshaped_train_labels, train_output[:, :self.rank_list_size], None)
            # have bug with new_ndcg
            # new_ndcg = ultra.utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(
            #     reshaped_train_labels, self.new_output[0][:, :self.rank_list_size], None)
            self.loss = 1.0 - previous_ndcg


            # Compute gradients
            # params = [p[1] for p in noise_list]
            # self.gradients = [p[0] * update_or_not for p in noise_list]
            
            # reshape from [rank_num, ?, ...] to [?, rank_num, ...]
            # reshaped_noise_lists = tf.einsum("ij...->ji...", tf.convert_to_tensor(noise_lists))
            # reshaped_param_lists = tf.einsum("ij...->ji...", tf.convert_to_tensor(param_lists))

            # params = [param_list[winner-1] for param_list, winner in zip(reshaped_param_lists, self.winners) if winner > 0]
            # self.gradients = [noise_list[winner-1] for i, winner in zip(reshaped_noise_lists, self.winners) if winner > 0]

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
                self.docid_inputs, train_output[:, :self.rank_list_size])
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
        print ("!!!!!!!!!!!!!", tf.shape(self.new_output))


        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.train_summary  # Summarize statistics.
            ]
            outputs = session.run(output_feed, input_feed)
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
