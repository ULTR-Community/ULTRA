from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
import ultra


class Linear(BaseRankingModel):
    """A linear model for learning to rank.

    This class implements a linear ranking model. It's essientially a logistic regression model.

    """

    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            initializer='None'                         # Set parameter initializer
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        if self.hparams.initializer == 'constant':
            self.initializer = tf.constant_initializer(0.001)

    def build(self, input_list, is_training=False):
        """ Create the model

        Args:
            input_list: (list<tf.Tensor>) A list of tensors containing the features
                        for a list of documents.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.

        """
        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            input_data = tf.concat(input_list, axis=0)
            output_data = tf.compat.v1.layers.batch_normalization(
                input_data, training=is_training, name="input_batch_normalization")
            output_sizes = [1]
            current_size = output_data.get_shape()[-1].value
            for j in range(len(output_sizes)):
                expand_W = tf.get_variable(
                    "linear_W_%d" % j, [current_size, output_sizes[j]])
                expand_b = tf.get_variable(
                    "linear_b_%d" % j, [output_sizes[j]])
                output_data = tf.nn.bias_add(
                    tf.matmul(output_data, expand_W), expand_b)
                output_data = tf.compat.v1.layers.batch_normalization(
                    output_data, training=is_training, name="batch_normalization_%d" % j)

            return tf.split(output_data, len(input_list), axis=0)

    def build_with_random_noise(
            self,
            input_list,
            noise_rate,
            is_training=False):
        """ Create the model

        Args:
            input_list: (list<tf.Tensor>) A list of tensors containing the features
                        for a list of documents.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
            A list of (tf.Tensor, tf.Tensor) containing the random noise and the parameters it is designed for.

        """
        noise_tensor_list = []
        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            input_data = tf.concat(input_list, axis=0)
            output_data = tf.compat.v1.layers.batch_normalization(
                input_data, training=is_training, name="input_batch_normalization")
            output_sizes = [1]
            current_size = output_data.get_shape()[-1].value
            for j in range(len(output_sizes)):
                original_W = tf.get_variable(
                    "linear_W_%d" % j, [current_size, output_sizes[j]])
                original_b = tf.get_variable(
                    "linear_b_%d" % j, [output_sizes[j]])
                # Create random noise
                random_W = tf.random.uniform(original_W.get_shape())
                random_b = tf.random.uniform(original_b.get_shape())
                noise_tensor_list.append((random_W, original_W))
                noise_tensor_list.append((random_b, original_b))
                expand_W = original_W + random_W * noise_rate
                expand_b = original_b + random_b * noise_rate
                # Run dnn
                output_data = tf.nn.bias_add(
                    tf.matmul(output_data, expand_W), expand_b)
                output_data = tf.compat.v1.layers.batch_normalization(
                    output_data, training=is_training, name="batch_normalization_%d" % j)

            return tf.split(output_data, len(input_list),
                            axis=0), noise_tensor_list
