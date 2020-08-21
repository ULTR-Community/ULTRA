"""Training and testing the Deep Listwise Context Model.

See the following paper for more information.

    * Qingyao Ai, Keping Bi, Jiafeng Guo, W. Bruce Croft. 2018. Learning a Deep Listwise Context Model for Ranking Refinement. In Proceedings of SIGIR '18
"""

# borrowed from DLCM model
# https://github.com/QingyaoAi/Deep-Listwise-Context-Model-for-Ranking-Refinement

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin
from ultra.ranking_model import BaseRankingModel
import ultra.utils
import copy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
# linear = rnn_cell_impl._linear  # pylint: disable=protected-access
linear = core_rnn_cell._linear


class DLCM(BaseRankingModel):
    """The Deep Listwise Context Model for learning to rank.

    This class implements the Deep Listwise Context Model (DLCM) for ranking.

    See the following paper for more information.

    * Qingyao Ai, Keping Bi, Jiafeng Guo, W. Bruce Croft. 2018. Learning a Deep Listwise Context Model for Ranking Refinement. In Proceedings of SIGIR '18

    """

    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Set the input sequences. "reverse","initial","random"
            input_sequence="initial",
            num_layers=1,                    # Number of layers in the model.
            # Number of heads in the attention strategy.
            num_heads=3,
            att_strategy='add',                # Select Attention strategy
            # Set to True for using LSTM cells instead of GRU cells.
            use_lstm=False,
        )
        print("building DLCM")
        self.hparams.parse(hparams_str)
        self.start_index = 0
        self.count = 1

        self.expand_embed_size = 50
        self.feed_previous = False
        self.output_projection = None
        scope = None
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
#         self.embeddings = tf.placeholder(tf.float32, shape=[None, embed_size], name="embeddings")
        self.target_labels = []
        self.target_weights = []
        self.target_initial_score = []
        with tf.variable_scope("embedding_rnn_seq2seq", reuse=tf.AUTO_REUSE):

            self.Layer_embedding = tf.keras.layers.LayerNormalization(
                name="embedding_norm")
            self.layer_norm_hidden = tf.keras.layers.LayerNormalization(
                name="layer_norm_state")
            self.layer_norm_final = tf.keras.layers.LayerNormalization(
                name="layer_norm_final")

    def _extract_argmax_and_embed(self, embedding, output_projection=None,
                                  update_embedding=True):
        """Get a loop_function that extracts the previous symbol and embeds it.

        Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

        Returns:
        A loop function.
        """

        def loop_function(prev, _):
            if output_projection is not None:
                prev = nn_ops.xw_plus_b(
                    prev, output_projection[0], output_projection[1])
            prev_symbol = math_ops.argmax(
                prev, 1) + tf.to_int64(self.batch_index_bias)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = tf.stop_gradient(emb_prev)
            return emb_prev

        return loop_function

    def rnn_decoder(self, encode_embed, attention_states, initial_state, cell,
                    num_heads=1, loop_function=None, dtype=dtypes.float32, scope=None,
                    initial_state_attention=False):
        """RNN decoder for the sequence-to-sequence model.

        """
        with tf.variable_scope(scope or "rnn_decoder"):
            batch_size = tf.shape(encode_embed[0])[0]  # Needed for reshaping.
            # number of output vector in sequence
            attn_length = attention_states.get_shape()[1].value
            # the dimension size of each output vector
            attn_size = attention_states.get_shape()[2].value
            # the dimension size of state vector
            state_size = initial_state.get_shape()[1].value
            print(batch_size, attn_length, attn_size, state_size,
                  "batch_size,attn_lengt,attn_size,state_size")
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to
            # reshape before.
            print(attention_states.get_shape(), "attention_states.get_shape()")
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            hidden_features2 = []
            v = []
            u = []
            linear_w = []
            linear_b = []
            abstract_w = []
            abstract_b = []
            abstract_layers = [int((attn_size + state_size) / (2 + 2 * i))
                               for i in xrange(2)] + [1]
            # Size of query vectors for attention.
            attention_vec_size = attn_size
            head_weights = []
            for a in xrange(num_heads):
                k = self.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
                hidden_features.append(nn_ops.conv2d(
                    hidden, k, [1, 1, 1, 1], "SAME"))  # [B,T,1,attn_vec_size]
                k2 = self.get_variable("AttnW2_%d" % a,
                                       [1, 1, attn_size, attention_vec_size])
                hidden_features2.append(nn_ops.conv2d(
                    hidden, k2, [1, 1, 1, 1], "SAME"))
                v.append(self.get_variable("AttnV_%d" % a,
                                           [attention_vec_size]))
                u.append(self.get_variable("AttnU_%d" % a,
                                           [attention_vec_size]))
                head_weights.append(self.get_variable(
                    "head_weight_%d" % a, [1]))
                current_layer_size = attn_size + state_size
                linear_w.append(self.get_variable("linearW_%d" % a,
                                                  [1, 1, current_layer_size, 1]))
                linear_b.append(self.get_variable("linearB_%d" % a,
                                                  [1]))
                abstract_w.append([])
                abstract_b.append([])
                for i in xrange(len(abstract_layers)):
                    layer_size = abstract_layers[i]
                    abstract_w[a].append(self.get_variable("Att_%d_layerW_%d" % (a, i),
                                                           [1, 1, current_layer_size, layer_size]))
                    abstract_b[a].append(self.get_variable("Att_%d_layerB_%d" % (a, i),
                                                           [layer_size]))
                    current_layer_size = layer_size

            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                ds = []  # Results of attention reads will be stored here.
                aw = []  # Attention weights will be stored here
                tiled_query = tf.tile(tf.reshape(
                    query, [-1, 1, 1, state_size]), [1, attn_length, 1, 1])
                print(hidden.get_shape(), "hidden.get_shape()")
                print(tiled_query.get_shape(), "tiled_query.get_shape()")
                concat_input = tf.concat(axis=3, values=[hidden, tiled_query])
                #concat_input = tf.concat(3, [hidden, hidden])
                for a in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % a):
                        s = None
                        if self.hparams.att_strategy == 'multi':
                            print('Attention: multiply')
                            y = linear(query, attention_vec_size, True)
                            y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                            # s = math_ops.reduce_sum(
                            # u[a] * math_ops.tanh(y * hidden_features[a]), [2,
                            # 3])
                            s = math_ops.reduce_sum(
                                hidden * math_ops.tanh(y), [2, 3])
                            # hidden_features[a] * math_ops.tanh(y), [2, 3])

                        elif self.hparams.att_strategy == 'multi_add':
                            print('Attention: multiply_add')
                            y = linear(query, attention_vec_size,
                                       True, scope='y')
                            y2 = linear(query, attention_vec_size,
                                        True, scope='y2')
                            y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                            y2 = tf.reshape(y2, [-1, 1, 1, attention_vec_size])
                            # s = math_ops.reduce_sum(
                            # u[a] * math_ops.tanh(y * hidden_features[a]), [2,
                            # 3])
                            s = math_ops.reduce_sum(
                                hidden * math_ops.tanh(y2), [2, 3])
                            s = s + math_ops.reduce_sum(
                                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

                        elif self.hparams.att_strategy == 'NTN':
                            print('Attention: NTN')
                            y = linear(query, attn_size, False)
                            y = tf.tile(tf.reshape(
                                y, [-1, 1, 1, attn_size]), [1, attn_length, 1, 1])
                            s = math_ops.reduce_sum(
                                hidden * y, [2, 3])  # bilnear
                            s = s + math_ops.reduce_sum(nn_ops.conv2d(concat_input, linear_w[a], [
                                                        1, 1, 1, 1], "SAME"), [2, 3])  # linear
                            s = s + linear_b[a]  # bias
                            # print(s.get_shape())
                            # s = tf.tanh(s) #non linear

                        elif self.hparams.att_strategy == 'elu':
                            print('Attention: elu')

                            cur_input = concat_input
                            # for i in xrange(len(abstract_layers)):
                            #    cur_input = tf.contrib.layers.fully_connected(cur_input, abstract_layers[i], activation_fn=tf.nn.elu)
                            for i in xrange(len(abstract_layers)):
                                cur_input = nn_ops.conv2d(
                                    cur_input, abstract_w[a][i], [1, 1, 1, 1], "SAME")
                                cur_input = cur_input + abstract_b[a][i]
                                cur_input = tf.nn.elu(cur_input)
                            s = math_ops.reduce_sum(cur_input, [2, 3])

                        else:
                            print('Attention: add')
                            y = linear(query, attention_vec_size, True)
                            y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                            s = math_ops.reduce_sum(
                                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

                        att = s * head_weights[a]  # nn_ops.softmax(s)
                        aw.append(att)
                        # Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum(
                            tf.reshape(att, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return aw, ds

            state = initial_state
            outputs = []
            prev = None
            batch_attn_size = tf.stack([batch_size, attn_size])
            batch_attw_size = tf.stack([batch_size, attn_length])
            attns = [tf.zeros(batch_attn_size, dtype=dtype)
                     for _ in xrange(num_heads)]
            attw = [1.0 / attn_length *
                    tf.ones(batch_attw_size, dtype=dtype) for _ in xrange(num_heads)]
            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])

            # Directly use previous state
            attw, attns = attention(initial_state)
            aw = math_ops.reduce_sum(attw, 0)
            output = tf.scalar_mul(1.0 / float(num_heads), aw)
            output = output - tf.reduce_min(output, 1, keep_dims=True)
            outputs.append(output)

        return outputs, state

    def embedding_rnn_decoder(self, initial_state, cell,
                              attention_states, encode_embed, num_heads=1,
                              output_projection=None,
                              feed_previous=False,
                              update_embedding_for_previous=True, scope=None):
        """RNN decoder with embedding and a pure-decoding option.

        """
        if output_projection is not None:
            proj_weights = ops.convert_to_tensor(output_projection[0],
                                                 dtype=dtypes.float32)
            proj_weights.get_shape().assert_is_compatible_with(
                [None, num_symbols])
            proj_biases = ops.convert_to_tensor(
                output_projection[1], dtype=dtypes.float32)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        with tf.variable_scope(scope or "embedding_rnn_decoder"):
            loop_function = self._extract_argmax_and_embed(
                encode_embed, output_projection,
                update_embedding_for_previous) if feed_previous else None
            # emb_inp = (
            #    embedding_ops.embedding_lookup(embeddings, i) for i in decoder_inputs)
            #emb_inp = decoder_embed
            return self.rnn_decoder(encode_embed, attention_states, initial_state, cell,
                                    num_heads=num_heads, loop_function=loop_function)

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """Create embedding RNN sequence-to-sequence model. No support for noisy parameters.

        Args:
            input_list: (list<tf.Tensor>) A list of tensors containing the features
                        for a list of documents.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        feed_previous = self.feed_previous
        embed_size = input_list[0].get_shape()[-1].value
        dtype = dtypes.float32
        output_projection = None
        list_size = len(input_list)  # len_seq
        with tf.variable_scope("cell", reuse=tf.AUTO_REUSE):
            single_cell = tf.contrib.rnn.GRUCell(
                embed_size + self.expand_embed_size)
            double_cell = tf.contrib.rnn.GRUCell(
                (embed_size + self.expand_embed_size) * 2)
            if self.hparams.use_lstm:
                single_cell = tf.contrib.rnn.BasicLSTMCell(
                    (embed_size + self.expand_embed_size))
                double_cell = tf.contrib.rnn.BasicLSTMCell(
                    (embed_size + self.expand_embed_size) * 2)
            cell = single_cell
            self.double_cell = double_cell
            self.cell = cell
            self.output_projection = output_projection

            if self.hparams.num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell] * self.hparams.num_layers)
                self.double_cell = tf.contrib.rnn.MultiRNNCell(
                    [double_cell] * self.hparams.num_layers)
        with tf.variable_scope(tf.get_variable_scope() or "embedding_rnn_seq2seq", reuse=tf.AUTO_REUSE):

            def abstract(input_data, index):
                reuse = None if index < 1 else True
                print(reuse, "reuse or not", tf.AUTO_REUSE, "tf.AUTO_REUSE")
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=tf.AUTO_REUSE):
                    output_data = input_data
                    output_sizes = [
                        int((embed_size + self.expand_embed_size) / 2), self.expand_embed_size]
                    current_size = embed_size
                    for i in xrange(2):
                        expand_W = self.get_variable(
                            "expand_W_%d" % i, [current_size, output_sizes[i]])
                        expand_b = self.get_variable(
                            "expand_b_%d" % i, [output_sizes[i]])
                        output_data = tf.nn.bias_add(
                            tf.matmul(output_data, expand_W), expand_b)
                        output_data = tf.nn.elu(output_data)
                        current_size = output_sizes[i]
                    return output_data
            for i in xrange(list_size):
                input_list[i] = self.Layer_embedding(input_list[i])
                if self.expand_embed_size > 0:
                    input_list[i] = tf.concat(axis=1, values=[input_list[i], abstract(
                        input_list[i], i)])  # [batch,feature_size+expand_embed_size]*len_seq
#             input_list= [tf.reshape(e, [1, -1,self.expand_embed_size+embed_size])
#                         for e in input_list]
#             input_list = tf.concat(axis=0, values=input_list)###[len_seq,batch,feature_size+expand_embed_size]
            # [len_seq,batch,feature_size+expand_embed_size]
            input_list = tf.stack(input_list, axis=0)
            enc_cell = copy.deepcopy(cell)
            ind = tf.range(0, list_size)
            print(self.hparams.input_sequence)
            if self.hparams.input_sequence == "reverse":
                ind = ind
            elif self.hparams.input_sequence == "initial":
                ind = tf.range(list_size - 1, -1, -1)
            elif self.hparams.input_sequence == "random":
                ind = tf.random.shuffle(ind)
            # [len_seq,batch,feature_size+expand_embed_size]
            input_list_input = tf.nn.embedding_lookup(input_list, ind)
            # [batch,feature_size+expand_embed_size]*len_seq
            input_list_input_list = tf.unstack(input_list_input, axis=0)
            encoder_outputs_some_order, encoder_state = tf.nn.static_rnn(
                enc_cell, input_list_input_list, dtype=dtype)
            ind_sort = tf.argsort(ind)  # find the order of sequence
#             ind_sort=tf.Print(tf.argsort(ind),[tf.argsort(ind),ind],"sequence")
            self.ind_sort = [ind_sort, ind]
            # [len_seq,batch,feature_size+expand_embed_size]
            encoder_outputs_some_order = tf.stack(
                encoder_outputs_some_order, axis=0)
#             encoder_outputs=[None]*list_size
#             for i in range(list_size):
#                 encoder_outputs[ind[i]]=encoder_outputs_some_order[i]## back to the order of initial list.
            # [len_seq,batch,feature_size+expand_embed_size]
            input_list_output = tf.nn.embedding_lookup(
                encoder_outputs_some_order, ind_sort)
            # [feature_size+expand_embed_size] *len_seq
            input_list_output_list = tf.unstack(input_list_output, axis=0)
            top_states = [tf.reshape(self.layer_norm_hidden(e), [-1, 1, cell.output_size])
                          for e in input_list_output_list]  # [batch,1,encoder_out]*len_seq
            encoder_state = self.layer_norm_final(
                encoder_state)  # [batch,encoder_state]
#             top_states = [tf.reshape(e, [-1, 1, cell.output_size])
#                         for e in encoder_outputs]  ##[]
#             encoder_state=encoder_state
            # for e in input_list]
            # [batch,len_seq,encoder_out]
            attention_states = tf.concat(axis=1, values=top_states)
#             print(attention_states.get_shape(),"attention_states.get_shape()")
            if isinstance(feed_previous, bool):
                outputs, state = self.embedding_rnn_decoder(
                    encoder_state, cell, attention_states, input_list,
                    num_heads=self.hparams.num_heads, output_projection=output_projection,
                    feed_previous=feed_previous)
                print(outputs[0].get_shape(), "outputs[0].get_shape()")
                return outputs[0]

            # If feed_previous is a Tensor, we construct 2 graphs and use cond.
            def decoder(feed_previous_bool):
                reuse = None if feed_previous_bool else True
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=reuse):
                    outputs, state = self.embedding_rnn_decoder(
                        encoder_state, cell, attention_states, input_list,
                        num_heads=self.hparams.num_heads, output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False)
                    return outputs + [state]

            outputs_and_state = control_flow_ops.cond(feed_previous,
                                                      lambda: decoder(True),
                                                      lambda: decoder(False))
            print(outputs[0].get_shape(), "outputs[0].get_shape()")
            return outputs_and_state[0]
