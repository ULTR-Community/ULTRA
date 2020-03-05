from __future__ import print_function
from __future__ import absolute_import
import os,sys
import tensorflow as tf
from .BasicRankingModel import BasicRankingModel
from .BasicRankingModel import ActivationFunctions


class DNN(BasicRankingModel):


    def __init__(self, hparams_str):
        """Create the network.
    
        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = tf.contrib.training.HParams(
            hidden_layer_sizes=[512, 256, 128],        # Number of neurons in each layer of a ranking_model. 
            activation_func='elu',                     # Type for activation function, which could be elu, relu, sigmoid, or tanh
            initializer='None'                         # Set parameter initializer
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.act_func = None
        if self.hparams.activation_func in BasicRankingModel.ACT_FUNC_DIC:
            self.act_func = BasicRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]
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
            output_data = tf.compat.v1.layers.batch_normalization(input_data, training=is_training, name="input_batch_normalization")
            output_sizes = self.hparams.hidden_layer_sizes + [1]
            current_size = output_data.get_shape()[-1].value
            for j in range(len(output_sizes)):
                expand_W = tf.get_variable("dnn_W_%d" % j, [current_size, output_sizes[j]]) 
                expand_b = tf.get_variable("dnn_b_%d" % j, [output_sizes[j]])
                output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                output_data = tf.compat.v1.layers.batch_normalization(output_data, training=is_training, name="batch_normalization_%d" % j)
                # Add activation if it is a hidden layer
                if j != len(output_sizes)-1:
                    output_data = self.act_func(output_data)
                current_size = output_sizes[j]
                
            return tf.split(output_data, len(input_list), axis=0)
    
    def build_with_random_noise(self, input_list, noise_rate, is_training=False):
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
            output_data = tf.compat.v1.layers.batch_normalization(input_data, training=is_training, name="input_batch_normalization")
            output_sizes = self.hparams.hidden_layer_sizes + [1]
            current_size = output_data.get_shape()[-1].value
            for j in range(len(output_sizes)):
                original_W = tf.get_variable("dnn_W_%d" % j, [current_size, output_sizes[j]]) 
                original_b = tf.get_variable("dnn_b_%d" % j, [output_sizes[j]])
                # Create random noise
                random_W = tf.random.uniform(original_W.get_shape())
                random_b = tf.random.uniform(original_b.get_shape())
                noise_tensor_list.append((random_W, original_W))
                noise_tensor_list.append((random_b, original_b))
                expand_W = original_W + random_W * noise_rate
                expand_b = original_b + random_b * noise_rate
                # Run dnn
                output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                output_data = tf.compat.v1.layers.batch_normalization(output_data, training=is_training, name="batch_normalization_%d" % j)
                # Add activation if it is a hidden layer
                if j != len(output_sizes)-1: 
                    output_data = self.act_func(output_data)
                current_size = output_sizes[j]
            return tf.split(output_data, len(input_list), axis=0), noise_tensor_list
