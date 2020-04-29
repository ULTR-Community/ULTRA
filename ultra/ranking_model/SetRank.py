"""Training and testing the SetRank model.

See the following paper for more information.
	
	* Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020. SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20
"""

from __future__ import print_function
from __future__ import absolute_import
import os,sys
import random
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
import ultra.utils
## Before changes we can get around 0.748 ndcg@10
## encodr part the transformer is borrowed from https://www.tensorflow.org/tutorials/text/transformer

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,training=True):
        batch_size = tf.shape(q)[0]
        print(q.get_shape(),"q.get_shape()")
        print("being called"+"*"*100)
        if q.get_shape()[-2]>10:
            collect=["eval"]
        else:
            collect=["train"]
#         q = self.wq(q)  # (batch_size, seq_len, d_model)
#         k = self.wk(k)  # (batch_size, seq_len, d_model)
#         v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = q  # (batch_size, seq_len, d_model)
        k = k  # (batch_size, seq_len, d_model)
        v = v  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,training,collect)  #(batch_size, num_heads, seq_len_q, depth)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask,training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

#         self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(maximum_position_encoding, 
#                                                 self.d_model)
        self.input_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.input_embedding=point_wise_feed_forward_network(d_model, dff)
        self.output_layer=point_wise_feed_forward_network(1, dff)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
#         self.softmax_output=tf.keras.layers.Softmax(1)
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        x=self.input_embedding(self.input_layer_norm(x))# (batch_size, input_seq_len, d_model)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
              x = self.enc_layers[i](x, training, mask)
        x=self.output_layer(x)
#         x=self.softmax_output(x)
        return x  # (batch_size, input_seq_len, 1)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
    tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
# def z_norm(a,axis=-1,scale=1):
#             return (a-tf.reduce_mean(a,axis=axis,keepdims=True))/(tf.math.reduce_std(a,axis=axis,keepdims=True)*scale+1e-10)
def scaled_dot_product_attention(q, k, v, mask=None,is_training=True,collect=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """
#         q_length=tf.norm(q,ord=2, axis=3,keepdims=True)
#         k_length=tf.norm(k,ord=2, axis=3,keepdims=True)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
#         matmul_qk=q_length*k_length
        print()
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.

        #       if is_training==True:
        #         collect_scope='train'
        #       else:
        #         collect_scope='eval'
        #       print(is_training)
#         select=tf.constant(["train","eval"])



        #       collect_scope_sum=tf.convert_to_tensor([collect_scope],tf.)
        #       summary_op = tf.summary.text("is_training", tf.strings.as_string(is_training),collections=['train'])
        #       summary_op1 = tf.summary.text("collect_scope", collect_scope,collections=['train'])  
#         def z_norm(a,axis=-1,scale=1):
#             return (a-tf.reduce_mean(a,axis=axis,keepdims=True))/(tf.math.reduce_std(a,axis=axis,keepdims=True)*scale+1e-10)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
#         d_seq = tf.cast(tf.shape(k)[-2], tf.float32)
#         #   scaled_attention_weights_withinq=tf.nn.softmax(tf.reduce_sum(scaled_attention_logits,axis=-1,keep_dims=True),axis=2)/ d_seq
#         #   scaled_attention_weights_betwq=tf.nn.softmax(tf.reduce_sum(scaled_attention_logits,axis=[2,3],keep_dims=True),axis=1)/ tf.math.square(d_seq)

# #         withinqbatch_k=tf.layers.BatchNormalization(2)
# #         betwqbatch_q=tf.layers.BatchNormalization(1)   

        last_dim=tf.reduce_mean(scaled_attention_logits,axis=-1,keep_dims=True)
        last2_dim=tf.reduce_mean(scaled_attention_logits,axis=[2,3],keep_dims=True)
        scaled_attention_weights_withinq=tf.nn.softmax(last_dim,axis=2)
        scaled_attention_weights_betwq=tf.nn.softmax(last2_dim,axis=1)
#         print(scaled_attention_weights_withinq.get_shape()\
#               ,"scaled_attention_weights_withinq.get_shape(),")
        weights_different_entry=tf.expand_dims(attention_weights[0,:,:,:],3)
        weights_different_q=tf.expand_dims(scaled_attention_weights_withinq[:,:,:,0],3)
        weights_different_Heads=tf.expand_dims(scaled_attention_weights_betwq[:,:,:,0],0)
#         attention_weights_horizontal=tf.nn.softmax(tf.reduce_sum(scaled_attention_logits,axis=[2],keep_dims=True),axis=1)
#         scaled_attention_weights_betwq_summary=tf.reduce_mean(attention_weights,axis=0)
#         scaled_attention_weights_withinq_1st=tf.expand_dims(attention_weights[0,:,:,:],axis=3)
#         attention_weights_horizontal=tf.reduce_sum(attention_weights,axis=2)
#         attention_weights_horizontal_row=tf.expand_dims(attention_weights_horizontal,axis=3)
#         scaled_attention_weights_betwq_1st=tf.expand_dims(scaled_attention_weights_betwq[:,:,:,0],axis=0)

        #       def for_train():
        #       tf.summary.histogram("train scaled_attention_weights_betwq_summary",scaled_attention_weights_betwq_summary,collections=collec)
        def for_train():
            tf.summary.image("weights_different_entry_[H,S,S,1]",\
                             weights_different_entry,max_outputs=8,collections=["train"])
            tf.summary.image("weights_different_q_[B,H,S,1]",\
                             weights_different_q,max_outputs=8,collections=["train"])
            tf.summary.image("weights_different_Heads_[1,B,H,1]",\
                             weights_different_Heads,max_outputs=8,collections=["train"])
            tf.summary.text("weights_first_portion_Heads_[1,B,H,1]",\
                            tf.as_string(weights_different_Heads[0,1,:,0]),collections=["train"])
            return True
        def for_eval():
            tf.summary.image("weights_different_entry_[H,S,S,1]",\
                             weights_different_entry,max_outputs=8,collections=["eval"])
            tf.summary.image("weights_different_q_[B,H,S,1]",\
                             weights_different_q,max_outputs=8,collections=["eval"])
            tf.summary.image("weights_different_Heads_[1,B,H,1]",\
                             weights_different_Heads,max_outputs=8,collections=["eval"])
            tf.summary.text("weights_first_portion_Heads_[1,B,H,1]",\
                            tf.as_string(weights_different_Heads[0,1,:,0]),collections=["eval"])
            return True
#         tf.cond(is_training,for_train,for_eval)
#         print(q.get_shape(),"q.get_shape()[-2]")
        if "eval" in collect:
            for_eval()
            print("image for eval")
        else:
            for_train()
            print("image for train")
        #       for i in range(tf.shape(k)[1]):
        #           tf.summary.scalar('attention_heads'+str(i), scaled_attention_weights_betwq_summary[i], collections=[collect_scope])
# #         attention_weights=attention_weights*scaled_attention_weights_withinq
# #         attention_weights=attention_weights*scaled_attention_weights_betwq
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights


# def scaled_dot_product_attention(q, k, v, mask):
#     """Calculate the attention weights.
#     q, k, v must have matching leading dimensions.
#     k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
#     The mask has different shapes depending on its type(padding or look ahead) 
#     but it must be broadcastable for addition.

#     Args:
#     q: query shape == (..., seq_len_q, depth)
#     k: key shape == (..., seq_len_k, depth)
#     v: value shape == (..., seq_len_v, depth_v)
#     mask: Float tensor with shape broadcastable 
#           to (..., seq_len_q, seq_len_k). Defaults to None.

#     Returns:
#     output, attention_weights
#     """

#     matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

#     # scale matmul_qk
#     dk = tf.cast(tf.shape(k)[-1], tf.float32)
#     scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

#     # add the mask to the scaled tensor.
#     if mask is not None:
#         scaled_attention_logits += (mask * -1e9)  

#     # softmax is normalized on the last axis (seq_len_k) so that the scores
#     # add up to 1.
#     attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

#     output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

#     return output, attention_weights


class SetRank(BaseRankingModel):
    """The SetRank model for learning to rank.

    This class implements the SetRank model for ranking.

    See the following paper for more information.
    
    * Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020. SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20

    """

    def __init__(self, hparams_str):
        """Create the network.
    
        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """
        print("build Transformer")
        self.hparams = ultra.utils.hparams.HParams(
            d_model=256, 
            num_heads=8,
            num_layers=2,
            diff=64,
            rate=0.0,
            initializer=None
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        if self.hparams.initializer == 'constant':
            self.initializer = tf.constant_initializer(0.001)
        with tf.variable_scope(tf.get_variable_scope()or "transformer",reuse=tf.AUTO_REUSE):
            self.Encoder_layer=Encoder(self.hparams.num_layers,self.hparams.d_model,\
                                       self.hparams.num_heads,self.hparams.diff,self.hparams.rate)
    def build(self, input_list, is_training=False):
        """ Create the model
        
        Args:
            input_list: (list<tf.Tensor>) A list of tensors containing the features 
                        for a list of documents.  list of [seq_len_q], each element [batch, feature_size] 
            is_training: (bool) A flag indicating whether the model is running in training mode.
        
        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.

        """
        with tf.variable_scope(tf.get_variable_scope() or "transformer",reuse=tf.AUTO_REUSE, initializer=self.initializer):
                sco_cur=tf.get_variable_scope()
                print(sco_cur.name,"sco_cur.name")
                mask=None
                batch_size = tf.shape(input_list[0])[0]
                feature_size = tf.shape(input_list[0])[1]
                list_size=len(input_list)
                ind=list(range(0,list_size))
                random.shuffle(ind)

#                 input_list=[input_list[i] for i in ind ]
                x= [tf.expand_dims(e,1)for e in input_list]
                
                x=tf.concat(axis=1,values=x)  ###[batch,len_seq,feature_size]
                x=self.Encoder_layer(x,is_training,mask)  ###[batch,len_seq,1]
                output=[]
                for i in range(list_size):
                    output.append(x[:,i,:])
#                 reind_output=[None]*list_size
#                 for i in range(list_size):
#                     reind_output[ind[i]]=output[i]  
#                 output=reind_output
        return output    ###[len_seq,batch,1]
## output should be a list of list_size while each element is [Batch,1]

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
                    output_data = tf.nn.elu(output_data)
                current_size = output_sizes[j]
            return tf.split(output_data, len(input_list), axis=0), noise_tensor_list
