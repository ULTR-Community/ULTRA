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
import tensorflow_ranking as tfr
from tensorflow import dtypes

from .BasicAlgorithm import BasicAlgorithm
sys.path.append("..")
import utils

class NavieAlgorithm(BasicAlgorithm):
    """The input_layer class that directly trains ranking models with clicks.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.
    
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build NavieAlgorithm')

        self.hparams = tf.contrib.training.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_cross_entropy',            # Select Loss function
            l2_loss=0.0,                    # Set strength for L2 regularization.
            grad_strategy='ada',            # Select gradient strategy
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
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                            name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                            name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)

        # Build model
        self.output = self.ranking_model(self.max_candidate_num, scope='ranking_model')
        
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels)) # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        pad_removed_output = self.remove_padding_for_metric_eval(self.docid_inputs, self.output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = utils.make_ranking_metric_fn(metric, topn)(reshaped_labels, pad_removed_output, None)
                tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['eval'])

        if not forward_only:
            # Build model
            self.rank_list_size = exp_settings['train_list_cutoff']
            train_output = self.ranking_model(self.rank_list_size, scope='ranking_model')
            train_labels = self.labels[:self.rank_list_size]
            reshaped_train_labels = tf.transpose(tf.convert_to_tensor(train_labels)) # reshape from [rank_list_size, ?] to [?, rank_list_size]

            self.loss = None
            if self.hparams.loss_func == 'sigmoid_cross_entropy':
                self.loss = self.sigmoid_loss(train_output, reshaped_train_labels)
            elif self.hparams.loss_func == 'pairwise_loss':
                self.loss = self.pairwise_loss(train_output, reshaped_train_labels)
            else:
                self.loss = self.softmax_loss(train_output, reshaped_train_labels)
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
            tf.summary.scalar('Loss', tf.reduce_mean(self.loss), collections=['train'])
            pad_removed_train_output = self.remove_padding_for_metric_eval(self.docid_inputs, train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = utils.make_ranking_metric_fn(metric, topn)(reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar('%s_%d' % (metric, topn), metric_value, collections=['train'])
            
        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def sigmoid_loss(self, output, labels, name=None):
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
        with tf.name_scope(name, "softmax_loss",[output]):
            label_dis = tf.math.minimum(labels, 1)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_dis, logits=output)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def pairwise_loss(self, output, labels, name=None):
        """Computes pairwise entropy loss.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "pairwise_loss",[output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            for i in range(len(sliced_output)):
                for j in range(i+1, len(sliced_output)):
                    cur_label_weight = tf.math.sign(sliced_label[i] - sliced_label[j])
                    cur_pair_loss = -tf.exp(sliced_output[i]) / (tf.exp(sliced_output[i]) + tf.exp(sliced_output[j]))
                    if loss == None:
                        loss = cur_label_weight * cur_pair_loss
                    loss += cur_label_weight * cur_pair_loss
        batch_size = tf.shape(labels[0])[0]
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32) #/ (tf.reduce_sum(propensity_weights)+1)


    def softmax_loss(self, output, labels, name=None):
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
