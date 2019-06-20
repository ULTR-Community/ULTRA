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

class PDGD(BasicAlgorithm):
    """The Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

    This class implements the Pairwise Differentiable Gradient Descent (PDGD) algorithm based on the input layer 
    feed. See the following paper for more information on the simulation data.
    
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

        self.hparams = tf.contrib.training.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            l2_loss=0.0,                    # Set strength for L2 regularization.
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings

        self.rank_list_size = data_set.rank_list_size
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
        # Build model
        if not forward_only:
            # Build training pair inputs only when it is training
            self.positive_docid_inputs = tf.placeholder(tf.int64, shape=[None], name="positive_docid_input")
            self.negative_docid_inputs = tf.placeholder(tf.int64, shape=[None], name="negative_docid_input")
            self.pair_weights = tf.placeholder(tf.float32, shape=[None], name="pair_weight")
            
            # Build ranking loss
            pair_scores = self.get_ranking_scores(
                [self.positive_docid_inputs, self.negative_docid_inputs], is_training=self.is_training, scope='ranking_model'
                )
            self.loss = self.pairwise_cross_entropy_loss(pair_scores[0], pair_scores[1])
            params = tf.trainable_variables()
            if self.hparams.l2_loss > 0:
                for p in params:
                    self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

            # Gradients and SGD update operation for training the model.
            opt = tf.train.AdagradOptimizer(self.hparams.learning_rate)
            self.gradients = tf.gradients(self.loss, params)
            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                     self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                             global_step=self.global_step)
                tf.summary.scalar('Gradient Norm', self.norm, collections=['train'])       
            else:
                self.norm = None 
                self.updates = opt.apply_gradients(zip(self.gradients, params),
                                             global_step=self.global_step)                 
            tf.summary.scalar('Learning Rate', self.learning_rate, collections=['train'])
            tf.summary.scalar('Loss', self.loss, collections=['train'])
            
        
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, None)
                tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['eval'])

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
            rank_outputs = session.run([self.output,self.eval_summary], input_feed)
            
            # reduce value to avoid numerical problems
            rank_outputs[0] = np.array(rank_outputs[0])
            for i in range(len(rank_outputs[0])):
                rank_outputs[0][i] = rank_outputs[0][i] - np.amax(rank_outputs[0][i])
            exp_ranking_scores = np.exp(rank_outputs[0])

            # Create training pairs based on the ranking scores and the labels
            positive_docids, negative_docids, pair_weights = [], [], []
            for i in range(len(input_feed[self.labels[0].name])):
                # Compute denominator
                p_r = np.ones(self.rank_list_size+1)
                denominators = np.zeros(self.rank_list_size+1)
                exp_scores = exp_ranking_scores[i]
                for j in range(self.rank_list_size):
                    idx = self.rank_list_size - 1 - j
                    if input_feed[self.docid_inputs[idx].name][i] < 0: # not a valid doc
                        continue
                    denominators[idx] = exp_scores[idx] + denominators[idx+1]
                    p_r[idx] = exp_scores[idx]/denominators[idx] * p_r[idx+1]
                
                # Generate pairs and compute weights
                for j in range(self.rank_list_size):
                    l = self.rank_list_size - 1 - j
                    if input_feed[self.labels[l].name][i] > 0: # a clicked doc
                        for k in range(l+1):
                            if input_feed[self.labels[k].name][i] == 0: # find a negative/unclicked doc
                                positive_docids.append(input_feed[self.docid_inputs[l].name][i])
                                negative_docids.append(input_feed[self.docid_inputs[k].name][i])
                                p_r_k_l = p_r[k] / p_r[min(l+1, self.rank_list_size-1)]
                                p_rs_k_l = 1.0
                                for x in range(l-k+1): # bug
                                    y = l-x
                                    p_rs_k_l *= exp_scores[y] / (denominators[y] - exp_scores[l] + exp_scores[k])
                                weight = p_rs_k_l / (p_r_k_l + p_rs_k_l)
                                pair_weights.append(weight) 
            input_feed[self.positive_docid_inputs.name] = positive_docids
            input_feed[self.negative_docid_inputs.name] = negative_docids
            input_feed[self.pair_weights.name] = pair_weights

            # Train the model
            input_feed[self.is_training.name] = True
            train_outputs = session.run([
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.train_summary # Summarize statistics.
                ], input_feed)
            summary = utils.merge_TFSummary([rank_outputs[-1], train_outputs[-1]], [0.5, 0.5])

            return train_outputs[1], rank_outputs, summary    # loss, no outputs, summary.
  
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary, # Summarize statistics.
                self.output   # Model outputs
            ]    
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
