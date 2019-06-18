from __future__ import print_function
from __future__ import absolute_import
import os,sys
import tensorflow as tf
from .BasicRankingModel import BasicRankingModel

class DNN(BasicRankingModel):
    def __init__(self, hparams_str):
        """Create the network.
    
        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = tf.contrib.training.HParams(
            hidden_layer_sizes=[512, 256, 128],        # Number of neurons in each layer of a ranking_model. 
        )
        self.hparams.parse(hparams_str)

    def build(self, input_list, is_training=False):
        """ Create the model
        
        Args:
            input_list: (list<tf.Tensor>) A list of tensors containing the features 
                        for a list of documents.
            is_training: (bool) A flag indicating whether the model is running in training mode.
        
        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.

        """
        output_list = []
        with tf.variable_scope(tf.get_variable_scope(),
                                            reuse=tf.AUTO_REUSE):
            for i in range(len(input_list)):
                output_data = tf.compat.v1.layers.batch_normalization(input_list[i], training=is_training)
                output_sizes = self.hparams.hidden_layer_sizes + [1]
                current_size = output_data.get_shape()[-1].value
                for i in range(len(output_sizes)):
                    expand_W = tf.get_variable("dnn_W_%d" % i, [current_size, output_sizes[i]]) 
                    expand_b = tf.get_variable("dnn_b_%d" % i, [output_sizes[i]])
                    output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                    output_data = tf.compat.v1.layers.batch_normalization(output_data, training=is_training)
                    output_data = tf.nn.elu(output_data)
                    current_size = output_sizes[i]
                output_list.append(output_data)
            return output_list