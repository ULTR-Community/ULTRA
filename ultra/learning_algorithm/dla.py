"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

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
import numpy as np
import tensorflow as tf

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def sigmoid_prob(logits):
    return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))


class DLA(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build DLA')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = tf.Variable(
                float(self.hparams.learning_rate), trainable=False)
        else:
            self.propensity_learning_rate = tf.Variable(
                float(self.hparams.propensity_learning_rate), trainable=False)
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

        # Select logits to prob function
        self.logits_to_prob = tf.nn.softmax
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.output = self.ranking_model(
            self.max_candidate_num, scope='ranking_model')
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)
        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels))
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
            self.propensity = self.DenoisingNet(
                self.rank_list_size, forward_only)
            train_labels = self.labels[:self.rank_list_size]

            print('Loss Function is ' + self.hparams.loss_func)
            # Select loss function
            self.loss_func = None
            if self.hparams.loss_func == 'sigmoid_loss':
                self.loss_func = self.sigmoid_loss_on_list
            elif self.hparams.loss_func == 'pairwise_loss':
                self.loss_func = self.pairwise_loss_on_list
            else:  # softmax loss without weighting
                self.loss_func = self.softmax_loss

            # Compute rank loss
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity))
            self.rank_loss = self.loss_func(
                train_output, reshaped_train_labels, self.propensity_weights)
            pw_list = tf.unstack(
                self.propensity_weights,
                axis=1)  # Compute propensity weights
            for i in range(len(pw_list)):
                tf.summary.scalar(
                    'Inverse Propensity weights %d' %
                    i, tf.reduce_mean(
                        pw_list[i]), collections=['train'])
            tf.summary.scalar(
                'Rank Loss',
                tf.reduce_mean(
                    self.rank_loss),
                collections=['train'])

            # Compute examination loss
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output))
            self.exam_loss = self.loss_func(
                self.propensity,
                reshaped_train_labels,
                self.relevance_weights)
            rw_list = tf.unstack(
                self.relevance_weights,
                axis=1)  # Compute propensity weights
            for i in range(len(rw_list)):
                tf.summary.scalar(
                    'Relevance weights %d' %
                    i, tf.reduce_mean(
                        rw_list[i]), collections=['train'])
            tf.summary.scalar(
                'Exam Loss',
                tf.reduce_mean(
                    self.exam_loss),
                collections=['train'])

            # Gradients and SGD update operation for training the model.
            self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            self.separate_gradient_update()

            tf.summary.scalar(
                'Gradient Norm',
                self.norm,
                collections=['train'])
            tf.summary.scalar(
                'Learning Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar(
                'Final Loss', tf.reduce_mean(
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

    def separate_gradient_update(self):
        denoise_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "denoising_model")
        ranking_model_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "ranking_model")

        if self.hparams.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        denoise_gradients = tf.gradients(self.exam_loss, denoise_params)
        ranking_model_gradients = tf.gradients(
            self.rank_loss, ranking_model_params)
        if self.hparams.max_gradient_norm > 0:
            denoise_gradients, denoise_norm = tf.clip_by_global_norm(denoise_gradients,
                                                                     self.hparams.max_gradient_norm)
            ranking_model_gradients, ranking_model_norm = tf.clip_by_global_norm(ranking_model_gradients,
                                                                                 self.hparams.max_gradient_norm * self.hparams.ranker_loss_weight)
        self.norm = tf.global_norm(denoise_gradients + ranking_model_gradients)

        opt_denoise = self.optimizer_func(self.propensity_learning_rate)
        opt_ranker = self.optimizer_func(self.learning_rate)

        denoise_updates = opt_denoise.apply_gradients(zip(denoise_gradients, denoise_params),
                                                      global_step=self.global_step)
        ranker_updates = opt_ranker.apply_gradients(
            zip(ranking_model_gradients, ranking_model_params))

        self.updates = tf.group(denoise_updates, ranker_updates)

    def DenoisingNet(self, list_size, forward_only=False, scope=None):
        with tf.variable_scope(scope or "denoising_model"):
            # If we are in testing, do not compute propensity
            if forward_only:
                return tf.ones_like(self.output)  # , tf.ones_like(self.output)
            input_vec_size = list_size

            def propensity_network(input_data, index):
                reuse = None if index < 1 else True
                propensity_initializer = tf.constant_initializer(
                    0.001) if self.hparams.constant_propensity_initialization else None
                with tf.variable_scope("propensity_network", initializer=propensity_initializer,
                                       reuse=reuse):
                    output_data = input_data
                    current_size = input_vec_size
                    output_sizes = [
                        #int((list_size+1)/2) + 1,
                        #int((list_size+1)/4) + 1,
                        1
                    ]
                    for i in range(len(output_sizes)):
                        expand_W = tf.get_variable(
                            "W_%d" % i, [current_size, output_sizes[i]])
                        expand_b = tf.get_variable(
                            "b_%d" % i, [output_sizes[i]])
                        output_data = tf.nn.bias_add(
                            tf.matmul(output_data, expand_W), expand_b)
                        output_data = tf.nn.elu(output_data)
                        current_size = output_sizes[i]
                    #expand_W = tf.get_variable("final_W", [current_size, 1])
                    #expand_b = tf.get_variable("final_b" , [1])
                    #output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                    return output_data

            output_propensity_list = []
            for i in range(list_size):
                # Add position information (one-hot vector)
                click_feature = [
                    tf.expand_dims(
                        tf.zeros_like(
                            self.labels[i]), -1) for _ in range(list_size)]
                click_feature[i] = tf.expand_dims(
                    tf.ones_like(self.labels[i]), -1)
                # Predict propensity with a simple network
                output_propensity_list.append(
                    propensity_network(
                        tf.concat(
                            click_feature, 1), i))

        return tf.concat(output_propensity_list, 1)

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
            return None, outputs[1], outputs[0]    # no loss, outputs, summary.

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = tf.unstack(
            propensity, axis=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = tf.stack(pw_list, axis=1)
        if self.hparams.max_propensity_weight > 0:
            propensity_weights = tf.clip_by_value(
                propensity_weights,
                clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights

    '''
    def click_weighted_softmax_cross_entropy_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_softmax_cross_entropy", [output]):
            label_dis = labels * propensity_weights / \
                tf.reduce_sum(labels * propensity_weights, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels * propensity_weights, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels * propensity_weights)

    def click_weighted_pairwise_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pairwise entropy loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
            (tf.Tensor) A tensor containing the propensity weights.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_pairwise_loss", [output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            sliced_propensity = tf.unstack(propensity_weights, axis=1)
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

    def click_weighted_log_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pointwise sigmoid loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_log_loss", [output]):
            click_prob = tf.sigmoid(output)
            loss = tf.losses.log_loss(labels, click_prob, propensity_weights)
        return loss
        '''
