"""Training and testing the Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Dueling Bandit Gradient Descent (DBGD) algorithm.
    
    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.
    
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

from . import ranking_model

from .BasicAlgorithm import BasicAlgorithm
sys.path.append("..")
import utils

class DBGD(BasicAlgorithm):
    """The Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

    This class implements the Dueling Bandit Gradient Descent (DBGD) algorithm based on the input layer 
    feed. See the following paper for more information on the simulation data.
    
    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.
    
    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.
    
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Dueling Bandit Gradient Descent (DBGD) algorithm.')

        self.hparams = tf.contrib.training.HParams(
            noise_rate=1,               # The update rate for randomly sampled weights.
            learning_rate=0.01,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            l2_loss=0.01,                    # Set strength for L2 regularization.
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        
        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
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
        self.output = tf.concat(self.get_ranking_scores(self.docid_inputs, is_training=self.is_training, scope='ranking_model'),1)
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = utils.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, None)
                tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['eval'])

        # Build model
        if not forward_only:
            self.rank_list_size = exp_settings['train_list_cutoff']
            train_output = tf.concat(self.get_ranking_scores(self.docid_inputs[:self.rank_list_size], is_training=self.is_training, scope='ranking_model'),1)
            train_labels = self.labels[:self.rank_list_size]
            # Create random gradients and apply it to get new ranking scores
            new_output_list, noise_list = self.get_ranking_scores_with_noise(self.docid_inputs[:self.rank_list_size], is_training=self.is_training, scope='ranking_model')
            
            # Compute NDCG for the old ranking scores and new ranking scores
            reshaped_train_labels = tf.transpose(tf.convert_to_tensor(train_labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
            self.new_output = tf.concat(new_output_list,1)
            previous_ndcg = utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(reshaped_train_labels, train_output, None)
            new_ndcg = utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(reshaped_train_labels, self.new_output, None)
            update_or_not = tf.ceil(new_ndcg - previous_ndcg)
            self.loss = 1.0 - new_ndcg

            # Compute gradients
            params = [p[1] for p in noise_list]
            self.gradients = [p[0] * update_or_not for p in noise_list]

            # Gradients and SGD update operation for training the model.
            opt = tf.train.AdagradOptimizer(self.hparams.learning_rate)
            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                     self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                             global_step=self.global_step)
                tf.summary.scalar('Gradient Norm', self.norm, collections=['train'])       
            else:
                self.norm = None 
                self.updates = opt.apply_gradients(zip(update_or_not * self.gradients, params),
                                             global_step=self.global_step)                 
            tf.summary.scalar('Learning Rate', self.learning_rate, collections=['train'])
            tf.summary.scalar('Loss', self.loss, collections=['train'])
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = utils.make_ranking_metric_fn(metric, topn)(reshaped_train_labels, train_output, None)
                    tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def get_ranking_scores(self, input_id_list, is_training=False, scope=None):
        """Run a step of the model feeding the given inputs.

        Args:
            input_id_list: (list<tf.Tensor>) A list of tensors containing document ids. 
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        with tf.variable_scope(scope or "ranking_model"):
            PAD_embed = tf.zeros([1,self.feature_size],dtype=tf.float32)
            letor_features = tf.concat(axis=0,values=[self.letor_features, PAD_embed])
            input_feature_list = []

            model = utils.find_class(self.exp_settings['ranking_model'])(self.exp_settings['ranking_model_hparams'])

            for i in range(len(input_id_list)):
                input_feature_list.append(tf.nn.embedding_lookup(letor_features, input_id_list[i]))
            return model.build(input_feature_list, is_training)
    
    def get_ranking_scores_with_noise(self, input_id_list, is_training=False, scope=None):
        """Run a step of the model feeding the given inputs.

        Args:
            input_id_list: (list<tf.Tensor>) A list of tensors containing document ids. 
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.
            A list of (tf.Tensor, tf.Tensor) containing the random noise and the parameters it is designed for.

        """
        with tf.variable_scope(scope or "ranking_model"):
            PAD_embed = tf.zeros([1,self.feature_size],dtype=tf.float32)
            letor_features = tf.concat(axis=0,values=[self.letor_features, PAD_embed])
            input_feature_list = []

            model = utils.find_class(self.exp_settings['ranking_model'])(self.exp_settings['ranking_model_hparams'])

            for i in range(len(input_id_list)):
                input_feature_list.append(tf.nn.embedding_lookup(letor_features, input_id_list[i]))
            return model.build_with_random_noise(input_feature_list, self.hparams.noise_rate, is_training)

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
            input_feed[self.is_training.name] = True
            output_feed = [
                            self.updates,    # Update Op that does SGD.
                            self.loss,    # Loss for this batch.
                            self.train_summary # Summarize statistics.
                            ] 
            outputs = session.run(output_feed, input_feed)
            return outputs[1], None, outputs[-1]    # loss, no outputs, summary.
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary, # Summarize statistics.
                self.output   # Model outputs
            ]    
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
