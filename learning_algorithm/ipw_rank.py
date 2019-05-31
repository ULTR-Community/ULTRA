"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.
    
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

def selu(x):
    with tf.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

class IPWrank(BasicAlgorithm):
    """This class implements the training and testing of the inverse propensity weighting 
        algorithm for unbiased learning to rank.

    See the following paper for more information on the dual learning algorithm.
    
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

        self.hparams = tf.contrib.training.HParams(
            propensity_estimator_type='utils.propensity_estimator.RandomizedPropensityEstimator',
            propensity_estimator_json='./example/PropensityEstimator/randomized_pbm_0.1_1.0_4_1.0.json', # the setting file for the predefined click models.
            learning_rate=0.5,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='click_weighted_softmax_cross_entropy',      # Select Loss function
            l2_loss=0.0,                    # Set strength for L2 regularization.
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.propensity_estimator = utils.find_class(self.hparams.propensity_estimator_type)(self.hparams.propensity_estimator_json)

        self.start_index = 0
        self.count = 1
        self.rank_list_size = data_set.rank_list_size
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        
        # Feeds for inputs.
        self.docid_inputs = [] # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size], 
                                name="letor_features") # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.propensity_weights = []
        for i in range(self.rank_list_size):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                            name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                            name="label{0}".format(i)))
            self.propensity_weights.append(tf.placeholder(tf.float32, shape=[None],
                                            name="propensity_weights{0}".format(i)))
        self.PAD_embed = tf.zeros([1,self.feature_size],dtype=tf.float32)
        for i in range(self.rank_list_size):
            tf.summary.scalar('Propensity weights %d' % i, tf.reduce_max(self.propensity_weights[i]), collections=['train'])

        # Build model
        self.output = self.ranking_model(forward_only)

        # Training outputs and losses.
        print('Loss Function is ' + self.hparams.loss_func)
        self.loss = None
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_propensity = tf.transpose(tf.convert_to_tensor(self.propensity_weights)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
        if self.hparams.loss_func == 'softmax':
            self.loss = self.softmax_loss(self.output, reshaped_labels, reshaped_propensity)
        elif self.hparams.loss_func == 'click_weighted_softmax_cross_entropy':
            self.loss = self.click_weighted_softmax_loss(self.output, reshaped_labels, reshaped_propensity)            
        else:
            self.loss = self.sigmoid_loss(self.output, reshaped_labels, reshaped_propensity)


        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
        if not forward_only:
            opt = tf.train.AdagradOptimizer(self.hparams.learning_rate)
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
            tf.summary.scalar('Learning Rate', self.learning_rate, collections=['train'])

            clipped_labels = tf.clip_by_value(reshaped_labels, clip_value_min=0, clip_value_max=1)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    list_weights = tf.reduce_mean(reshaped_propensity * clipped_labels, axis=1, keep_dims=True)
                    metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, list_weights)
                    tf.summary.scalar('Weighted_%s@%d' % (metric.upper(), topn), metric_value, collections=['train'])
        
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, None)
                tf.summary.scalar('%s@%d' % (metric.upper(), topn), metric_value, collections=['train', 'eval'])

        tf.summary.scalar('Loss', tf.reduce_mean(self.loss), collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())


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
        # compute propensity weights for the input data.
        for l in range(self.rank_list_size):
            input_feed[self.propensity_weights[l].name] = []
        for i in range(len(input_feed[self.labels[0].name])):
            click_list = [input_feed[self.labels[l].name][i] for l in range(self.rank_list_size)]
            pw_list = self.propensity_estimator.getPropensityForOneList(click_list)
            for l in range(self.rank_list_size):
                input_feed[self.propensity_weights[l].name].append(pw_list[l])
        
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
        with tf.name_scope(name, "sigmoid_loss",[output]):
            original_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)
            loss = original_loss * propensity
        batch_size = tf.shape(labels[0])[0]
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32) #/ (tf.reduce_sum(propensity_weights)+1)

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
        with tf.name_scope(name, "softmax_loss",[output]):
            label_dis = labels / tf.reduce_sum(labels, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(labels, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels)

    
    def click_weighted_softmax_loss(self, output, labels, propensity, name=None):
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
        with tf.name_scope(name, "ipw_softmax_loss",[output]):
            label_dis = labels * propensity / tf.reduce_sum(labels * propensity, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(labels * propensity, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels * propensity)



