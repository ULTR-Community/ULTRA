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
import tensorflow_ranking as tfr
import copy
import itertools
from six.moves import zip
from tensorflow import dtypes

from . import ranking_model
from . import metrics
from .BasicAlgorithm import BasicAlgorithm
sys.path.append("..")
import utils


def sigmoid_prob(logits):
    return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))

class DLA(BasicAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer 
    feed. See the following paper for more information on the simulation data.
    
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

        self.hparams = tf.contrib.training.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='click_weighted_softmax_cross_entropy',            # Select Loss function
            logits_to_prob='softmax',        # the function used to convert logits to probability distributions
            ranker_learning_rate=-1.0,         # The learning rate for ranker (-1 means same with learning_rate).
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            l2_loss=0.0,                    # Set strength for L2 regularization.
            grad_strategy='ada',            # Select gradient strategy
            relevance_category_num=5,        # Select the number of relevance category
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings

        self.start_index = 0
        self.count = 1
        self.rank_list_size = data_set.rank_list_size
        self.feature_size = data_set.feature_size
        if self.hparams.ranker_learning_rate < 0:
            self.ranker_learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        else:
            self.ranker_learning_rate = tf.Variable(float(self.hparams.ranker_learning_rate), trainable=False)
        self.learning_rate = self.ranker_learning_rate
        
        # Feeds for inputs.
        self.docid_inputs = [] # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size], 
                                name="letor_features") # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.rank_list_size):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                            name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                            name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)

        # Select logits to prob function
        self.logits_to_prob = tf.nn.softmax
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        # Build model
        self.output = self.ranking_model(forward_only)
        self.propensity = self.DenoisingNet(forward_only)

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'click_weighted_softmax_cross_entropy':
            self.loss_func = self.click_weighted_softmax_cross_entropy_loss
        elif self.hparams.loss_func == 'click_weighted_log_loss':
            self.loss_func = self.click_weighted_log_loss
        else: # softmax loss without weighting
            self.loss_func = self.softmax_loss

        # Compute rank loss
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
        self.rank_loss, self.propensity_weights = self.loss_func(self.output, reshaped_labels, self.propensity)
        pw_list = tf.split(self.propensity_weights, self.rank_list_size, 1) # Compute propensity weights
        for i in range(self.rank_list_size):
            tf.summary.scalar('Avg Propensity weights %d' % i, tf.reduce_mean(pw_list[i]), collections=['train'])
        tf.summary.scalar('Rank Loss', tf.reduce_mean(self.rank_loss), collections=['train'])

        # Compute examination loss
        self.exam_loss, self.relevance_weights = self.loss_func(self.propensity, reshaped_labels, self.output)
        rw_list = tf.split(self.relevance_weights, self.rank_list_size, 1) # Compute propensity weights
        for i in range(self.rank_list_size):
            tf.summary.scalar('Avg Relevance weights %d' % i, tf.reduce_mean(rw_list[i]), collections=['train'])
        tf.summary.scalar('Exam Loss', tf.reduce_mean(self.exam_loss), collections=['train'])
        
        # Gradients and SGD update operation for training the model.
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        if not forward_only:
            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            self.separate_gradient_update()
        
            tf.summary.scalar('Gradient Norm', self.norm, collections=['train'])
            tf.summary.scalar('Learning Rate', self.ranker_learning_rate, collections=['train'])
            tf.summary.scalar('Final Loss', tf.reduce_mean(self.loss), collections=['train'])
        
            clipped_labels = tf.clip_by_value(reshaped_labels, clip_value_min=0, clip_value_max=1)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    list_weights = tf.reduce_mean(self.propensity_weights * clipped_labels, axis=1, keep_dims=True)
                    metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, list_weights)
                    tf.summary.scalar('Weighted_%s_%d' % (metric, topn), metric_value, collections=['train'])
        
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, None)
                tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['train', 'eval'])


        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def separate_gradient_update(self):
        denoise_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "denoising_model")
        ranking_model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ranking_model")

        if self.hparams.l2_loss > 0:
            for p in denoise_params:
                self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        denoise_gradients = tf.gradients(self.exam_loss, denoise_params)
        ranking_model_gradients = tf.gradients(self.rank_loss, ranking_model_params)
        if self.hparams.max_gradient_norm > 0:
            denoise_gradients, denoise_norm = tf.clip_by_global_norm(denoise_gradients,
                                                                     self.hparams.max_gradient_norm)
            ranking_model_gradients, ranking_model_norm = tf.clip_by_global_norm(ranking_model_gradients,
                                                                     self.hparams.max_gradient_norm * self.hparams.ranker_loss_weight)
        self.norm = tf.global_norm(denoise_gradients + ranking_model_gradients)

        opt_denoise = self.optimizer_func(self.hparams.learning_rate)
        opt_ranker = self.optimizer_func(self.ranker_learning_rate)

        denoise_updates = opt_denoise.apply_gradients(zip(denoise_gradients, denoise_params),
                                            global_step=self.global_step)
        ranker_updates = opt_ranker.apply_gradients(zip(ranking_model_gradients, ranking_model_params))

        self.updates = tf.group(denoise_updates, ranker_updates)

    def ranking_model(self, forward_only=False, scope=None):
        with tf.variable_scope(scope or "ranking_model"):
            PAD_embed = tf.zeros([1,self.feature_size],dtype=tf.float32)
            letor_features = tf.concat(axis=0,values=[self.letor_features, PAD_embed])
            input_feature_list = []
            output_scores = []

            model = utils.find_class(self.exp_settings['ranking_model'])(self.exp_settings['ranking_model_hparams'])

            for i in range(self.rank_list_size):
                input_feature_list.append(tf.nn.embedding_lookup(letor_features, self.docid_inputs[i]))
            output_scores = model.build(input_feature_list)

            return tf.concat(output_scores,1)

    def DenoisingNet(self, forward_only=False, scope=None):
        with tf.variable_scope(scope or "denoising_model"):
            # If we are in testing, do not compute propensity
            if forward_only:
                return tf.ones_like(self.output)#, tf.ones_like(self.output)
            input_vec_size = self.rank_list_size

            def propensity_network(input_data, index):
                reuse = None if index < 1 else True
                with tf.variable_scope("propensity_network",
                                                 reuse=reuse):
                    output_data = input_data
                    current_size = input_vec_size
                    output_sizes = [
                        int((self.hparams.relevance_category_num+self.rank_list_size+1)/2), 
                        int((self.hparams.relevance_category_num+self.rank_list_size+1)/4),
                    ]
                    for i in range(len(output_sizes)):
                        expand_W = tf.get_variable("W_%d" % i, [current_size, output_sizes[i]])
                        expand_b = tf.get_variable("b_%d" % i, [output_sizes[i]])
                        output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                        output_data = tf.nn.elu(output_data)
                        current_size = output_sizes[i]
                    expand_W = tf.get_variable("final_W", [current_size, 1])
                    expand_b = tf.get_variable("final_b" , [1])
                    output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                    return output_data

            output_propensity_list = []
            for i in range(self.rank_list_size):
                # Add position information (one-hot vector)
                click_feature = [tf.expand_dims(tf.zeros_like(self.labels[i]) , -1) for _ in range(self.rank_list_size)]
                click_feature[i] = tf.expand_dims(tf.ones_like(self.labels[i]) , -1)
                # Predict propensity with a simple network
                output_propensity_list.append(propensity_network(tf.concat(click_feature, 1), i))
            
        return tf.concat(output_propensity_list,1)

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
            output_feed = [self.updates,    # Update Op that does SGD.
                            self.loss,    # Loss for this batch.
                            self.train_summary # Summarize statistics.
                            ]    
        else:
            output_feed = [self.loss, # Loss for this batch.
                        self.eval_summary, # Summarize statistics.
                        self.output   # Model outputs
            ]    

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], None, outputs[-1]    # loss, no outputs, summary.
        else:
            return outputs[0], outputs[2], outputs[1]    # loss, outputs, summary.

    def softmax_loss(self, output, labels, propensity=None, name=None):
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
        with tf.name_scope(name, "softmax_loss",[output]):
            propensity_weights = tf.ones_like(output)
            label_dis = labels / tf.reduce_sum(labels, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(labels, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels), propensity_weights

    def click_weighted_softmax_cross_entropy_loss(self, output, labels, propensity, name=None):
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
        with tf.name_scope(name, "click_softmax_cross_entropy",[output]):
            propensity_list = tf.split(self.logits_to_prob(propensity), self.rank_list_size, 1) # Compute propensity weights
            pw_list = []
            for i in range(self.rank_list_size):
                pw_i = propensity_list[0] / propensity_list[i]
                pw_list.append(pw_i)
            propensity_weights = tf.concat(pw_list, 1)
            label_dis = labels*propensity_weights / tf.reduce_sum(labels*propensity_weights, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(labels*propensity_weights, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels*propensity_weights), propensity_weights

    def click_weighted_log_loss(self, output, labels, propensity, name=None):
        """Computes pointwise sigmoid loss with propensity weighting.

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
        with tf.name_scope(name, "click_weighted_log_loss",[output]):
            propensity_list = tf.split(self.logits_to_prob(propensity), self.rank_list_size, 1) # Compute propensity weights
            pw_list = []
            for i in range(self.rank_list_size):
                pw_i = propensity_list[0] / propensity_list[i]
                pw_list.append(pw_i)
            propensity_weights = tf.concat(pw_list, 1)
            click_prob = tf.sigmoid(output)
            loss = tf.losses.log_loss(labels, click_prob, propensity_weights)
        return loss, propensity_weights
