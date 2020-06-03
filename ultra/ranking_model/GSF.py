from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
from ultra.ranking_model import ActivationFunctions
import ultra.utils


class GSF(BaseRankingModel):
    """The Groupwise Scoring Function (with no approaximation) for learning to rank.

    This class implements The Groupwise Scoring Function (GSF) based on multi-layer perceptron networks.

    See the following paper for more information.

    * Qingyao Ai, Xuanhui Wang, Sebastian Bruch, Nadav Golbandi, Michael Bendersky, Marc Najork. 2019. Learning Groupwise Scoring Functions Using Deep Neural Networks. In Proceedings of ICTIR '19

    """

    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Number of neurons in each layer of a ranking_model.
            hidden_layer_sizes=[512, 256, 128],
            # Number of inputs for each groupwise function.
            group_size=2,
            # Type for activation function, which could be elu, relu, sigmoid,
            # or tanh
            activation_func='elu',
            initializer='None',                         # Set parameter initializer
            norm="layer"                                # Set the default normalization
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.act_func = None
        self.layer_norm = None

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

        self.model_parameters = {}

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the GSF model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            input_data = tf.concat(input_list, axis=0)
            output_data = input_data
            input_data_list = tf.split(output_data, len(input_list), axis=0)
            list_size = len(input_list)

            # Define groupwise scoring functions with DNN
            output_sizes = self.hparams.hidden_layer_sizes + \
                [self.hparams.group_size]

            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                self.layer_norm = []
                for j in range(len(output_sizes)):
                    self.layer_norm.append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                        name="layer_norm_%d" % j))

            def dnn(x):
                current_size = x.get_shape()[-1].value
                for j in range(len(output_sizes)):
                    if self.layer_norm is not None:
                        x = self.layer_norm[j](x, training=is_training)
                    expand_W = self.get_variable(
                        "dnn_W_%d" % j, [current_size, output_sizes[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                    expand_b = self.get_variable(
                        "dnn_b_%d" % j, [output_sizes[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                    x = tf.nn.bias_add(tf.matmul(x, expand_W), expand_b)
                    # Add activation if it is a hidden layer
                    if j != len(output_sizes) - 1:
                        x = self.act_func(x)
                    current_size = output_sizes[j]
                return tf.split(x, self.hparams.group_size, axis=1)

            # Apply groupwise scoring functions
            def get_possible_group(group, group_list):
                if len(group) == self.hparams.group_size:
                    group_list.append(group)
                    return
                else:
                    for i in range(list_size):
                        get_possible_group(group + [i], group_list)

            output_data_list = [0 for _ in range(list_size)]
            group_list = []
            get_possible_group([], group_list)
            for group in group_list:
                group_input = tf.concat([input_data_list[idx]
                                         for idx in group], axis=1)
                group_score_list = dnn(group_input)
                for i in range(self.hparams.group_size):
                    output_data_list[group[i]] += group_score_list[i]

            return output_data_list
