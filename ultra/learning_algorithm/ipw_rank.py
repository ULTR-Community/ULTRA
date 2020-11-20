"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the inverse propensity weighting algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

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


def selu(x):
    with tf.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


class IPWrank(BaseAlgorithm):
    """The Inverse Propensity Weighting algorithm for unbiased learning to rank.

    This class implements the training and testing of the Inverse Propensity Weighting algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """

        self.hparams = ultra.utils.hparams.HParams(
            propensity_estimator_type='ultra.utils.propensity_estimator.RandomizedPropensityEstimator',
            # the setting file for the predefined click models.
            propensity_estimator_json='./example/PropensityEstimator/randomized_pbm_0.1_1.0_4_1.0.json',
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',      # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        self.model = None
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.propensity_estimator = ultra.utils.find_class(
            self.hparams.propensity_estimator_type)(
            self.hparams.propensity_estimator_json)

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

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
        self.PAD_embed = tf.zeros([1, self.feature_size], dtype=tf.float32)

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

        # Gradients and SGD update operation for training the model.
        if not forward_only:
            # Training outputs and losses.
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            train_labels = self.labels[:self.rank_list_size]
            self.propensity_weights = []
            for i in range(self.rank_list_size):
                self.propensity_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                              name="propensity_weights{0}".format(i)))
                tf.summary.scalar(
                    'Propensity weights %d' %
                    i,
                    tf.reduce_max(
                        self.propensity_weights[i]),
                    collections=['train'])

            print('Loss Function is ' + self.hparams.loss_func)
            self.loss = None
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_propensity = tf.transpose(
                tf.convert_to_tensor(self.propensity_weights))
            if self.hparams.loss_func == 'sigmoid_loss':
                self.loss = self.sigmoid_loss_on_list(
                    train_output, reshaped_train_labels, reshaped_propensity)
            elif self.hparams.loss_func == 'pairwise_loss':
                self.loss = self.pairwise_loss_on_list(
                    train_output, reshaped_train_labels, reshaped_propensity)
            else:
                self.loss = self.softmax_loss(
                    train_output, reshaped_train_labels, reshaped_propensity)

            params = tf.trainable_variables()
            if self.hparams.l2_loss > 0:
                for p in params:
                    self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            opt = self.optimizer_func(self.hparams.learning_rate)
            self.gradients = tf.gradients(self.loss, params)
            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                           self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                                   global_step=self.global_step)
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
                        reshaped_propensity * clipped_labels, axis=1, keep_dims=True)
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
            # compute propensity weights for the input data.
            for l in range(self.rank_list_size):
                input_feed[self.propensity_weights[l].name] = []
            for i in range(len(input_feed[self.labels[0].name])):
                click_list = [input_feed[self.labels[l].name][i]
                              for l in range(self.rank_list_size)]
                pw_list = self.propensity_estimator.getPropensityForOneList(
                    click_list)
                for l in range(self.rank_list_size):
                    input_feed[self.propensity_weights[l].name].append(
                        pw_list[l])

            input_feed[self.is_training.name] = True
            output_feed = [self.updates,    # Update Op that does SGD.
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
    '''
    def sigmoid_loss(self, output, labels, propensity, name=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity: No use.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "sigmoid_loss", [output]):
            original_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=output)
            loss = original_loss * propensity
        batch_size = tf.shape(labels[0])[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32)

    def click_weighted_pairwise_loss(
            self, output, labels, propensity, name=None):
        """Computes pairwise entropy loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_pairwise_loss", [output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            sliced_propensity = tf.unstack(propensity, axis=1)
            for i in range(len(sliced_output)):
                for j in range(i + 1, len(sliced_output)):
                    cur_label_weight = tf.math.sign(
                        sliced_label[i] - sliced_label[j])
                    cur_propensity = sliced_propensity[i] * \
                        sliced_label[i] + \
                        sliced_propensity[j] * sliced_label[j]
                    cur_pair_loss = - \
                        tf.exp(
                            sliced_output[i]) / (tf.exp(sliced_output[i]) + tf.exp(sliced_output[j]))
                    if loss is None:
                        loss = cur_label_weight * cur_pair_loss * cur_propensity
                    loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = tf.shape(labels[0])[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32)

    def softmax_loss(self, output, labels, propensity, name=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity: No use.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "softmax_loss", [output]):
            label_dis = labels / tf.reduce_sum(labels, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels)

    def click_weighted_softmax_loss(
            self, output, labels, propensity, name=None):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "ipw_softmax_loss", [output]):
            label_dis = labels * propensity / \
                tf.reduce_sum(labels * propensity, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels * propensity, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels * propensity)
        '''