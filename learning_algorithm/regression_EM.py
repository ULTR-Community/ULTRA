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


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return tf.ceil(probs - tf.random_uniform(tf.shape(probs)))

class RegressionEM(BasicAlgorithm):
    """The regression-based EM algorithm for unbiased learning to rank.

    This class implements the regression-based EM algorithm based on the input layer 
    feed. See the following paper for more information on the simulation data.
    
    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.
    
    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.
    
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Regression-based EM algorithm.')

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

        # Build model
        self.output = self.ranking_model(forward_only)
        
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
        if not forward_only:
            # Build EM graph only when it is training
            self.propensity = tf.Variable(tf.ones([1, self.rank_list_size]) / self.rank_list_size, trainable=False)

            self.splitted_propensity = tf.split(self.propensity, self.rank_list_size, axis=1)
            for i in range(self.rank_list_size):
                tf.summary.scalar('Examination Probability %d' % i, tf.reduce_max(self.splitted_propensity[i]), collections=['train'])

            # Conduct estimation step.
            gamma = tf.sigmoid(self.output)
            reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]
            p_e1_r1_c1 = 1
            p_e1_r0_c0 = self.propensity * (1 - gamma) / (1 - self.propensity * gamma)
            p_e0_r1_c0 = (1 - self.propensity) * gamma / (1 - self.propensity * gamma)
            p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / (1 - self.propensity * gamma)
            p_e1 = p_e1_r0_c0 + p_e1_r1_c1
            p_r1 = reshaped_labels + (1-reshaped_labels) * p_e0_r1_c0

            # Conduct maximization step
            self.update_propensity_op =  self.propensity.assign(
                tf.reduce_mean(
                    reshaped_labels + (1-reshaped_labels) * p_e1_r0_c0, axis=0, keep_dims=True
                )# / tf.reduce_sum(reshaped_labels, axis=0)
            )

            # Get Bernoulli samples and compute rank loss
            self.ranker_labels = get_bernoulli_sample(p_r1)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ranker_labels, logits=self.output)
            )
            reshaped_propensity = self.propensity
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
            else:
                self.norm = None 
                self.updates = opt.apply_gradients(zip(self.gradients, params),
                                             global_step=self.global_step)
            tf.summary.scalar('Learning Rate', self.learning_rate, collections=['train'])
            tf.summary.scalar('Loss', tf.reduce_mean(self.loss), collections=['train'])
            clipped_labels = tf.clip_by_value(reshaped_labels, clip_value_min=0, clip_value_max=1)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    list_weights = tf.reduce_mean(reshaped_propensity * clipped_labels, axis=1, keep_dims=True)
                    metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, list_weights)
                    tf.summary.scalar('Weighted_%s_%d' % (metric, topn), metric_value, collections=['train'])

        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = metrics.make_ranking_metric_fn(metric, topn)(reshaped_labels, self.output, None)
                tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['train', 'eval'])

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
            output_scores = model.build(input_feature_list, is_training=self.is_training)

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
        
        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                            self.updates,    # Update Op that does SGD.
                            self.loss,    # Loss for this batch.
                            self.update_propensity_op,
                            self.train_summary # Summarize statistics.
                            ]    
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                        self.eval_summary, # Summarize statistics.
                        self.output   # Model outputs
            ]    

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], None, outputs[-1]    # loss, no outputs, summary.
        else:
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
