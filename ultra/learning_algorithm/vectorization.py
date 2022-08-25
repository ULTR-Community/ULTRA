"""Training and testing the Vectorization for unbiased learning to rank.

See the following paper for more information on the Vectorization.

    * Mouxiang Chen, Chenghao Liu, Zemin Liu, Jianling Sun. 2022. Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank. In Proceedings of SIGKDD '22.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import zip

import ultra.utils
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm


class Vectorization(BaseAlgorithm):
    """Vectorization for unbiased learning to rank.

    See the following paper for more information on the Vectorization.

    * Mouxiang Chen, Chenghao Liu, Zemin Liu, Jianling Sun. 2022. Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank. In Proceedings of SIGKDD '22.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Vectorization')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,  # Learning rate.
            max_gradient_norm=5.0,  # Clip gradients to this norm.
            l2_loss=0.0,  # Set strength for L2 regularization.
            grad_strategy='ada',  # Optimizer
            dimension=3,  # Vector dimension
            pretrain_ranker_step=500,
            # The step for freezing the observation model and the base model
            prob_l2_loss=0.001,  # L2 regularization for the base model
            affine=0
            # 0/1, indicates whether to run in the Affine mode (Ali Vardasbi,
            # Harrie Oosterhuis, and Maarten de Rijke. When Inverse Propensity
            # Scoring does not Work: Affine Corrections for Unbiased Learning
            # to Rank. CIKM 2020)
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        print("hparam", self.hparams.to_json())
        self.exp_settings = exp_settings
        if self.exp_settings['ranking_model_hparams'].strip() != '':
            self.exp_settings['ranking_model_hparams'] += ","
        self.exp_settings['ranking_model_hparams'] += (
            "output_size=" + str(self.hparams.dimension))
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)
        self.forward_only = forward_only
        self.propensity_model = None

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)

        self.output_tuple = self.build_models(self.max_candidate_num,
                                              is_predict=True)
        self.output = self.estimate_output(self.output_tuple)

        print([v.name for v in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)])

        if not forward_only:
            # Build model
            self.rank_list_size = exp_settings['selection_bias_cutoff']

            rel_vector, propensity_vector, base_vector = self.build_models(
                self.rank_list_size)  # [B, S, d]
            click = self.combine_vector(
                rel_vector, propensity_vector)  # [B, S]
            trained_labels = self.labels[:self.rank_list_size]  # (S, B)
            trained_labels = tf.transpose(tf.stack(trained_labels))  # (B, S)
            self.supervise_loss = self.softmax_loss(click, trained_labels)
            self.base_vector_loss = self.build_observation_density_loss(
                propensity_vector)
            self.loss = self.supervise_loss + self.base_vector_loss

            self.scalar(self.supervise_loss, "supervise_loss")

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            self.build_update(self.loss)

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def evaluate(self, output):
        # output: (B, T)
        # label: (B, T)
        label = self.labels
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, output)
        reshaped_labels = tf.transpose(tf.convert_to_tensor(label))
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                item_name = '%s_%d' % (metric, topn)
                tf.summary.scalar(
                    item_name, metric_value, collections=['eval'])

    def combine_vector(self, v1, v2, keepdims=False):
        return tf.reduce_sum(v1 * v2, axis=-1, keepdims=keepdims)

    def estimate_output(self, output_tuple):

        relevance, _, base_vector = output_tuple  # (B, T, d)

        if self.hparams.affine == 1:
            output = relevance[:, :, 0]
        else:
            output = self.combine_vector(relevance, base_vector)
        self.evaluate(output)

        return output

    def build_update(self, loss):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if self.hparams.l2_loss > 0:
            for p in params:
                loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

        gradients = tf.gradients(loss, params)
        if self.hparams.max_gradient_norm > 0:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.hparams.max_gradient_norm)
        self.norm = tf.global_norm(gradients)

        optimizer = self.optimizer_func(self.learning_rate)
        self.update = optimizer.apply_gradients(zip(gradients, params),
                                                global_step=self.global_step)

    def build_models(self, T, is_predict=False, **kwargs):
        # Return (B, T, d)
        x = self.get_ranking_scores(self.docid_inputs[:T], self.is_training,
                                    **kwargs)  # (T, B, S_max)
        relevance = tf.stack(x)  # (T, B, xxx)
        if len(relevance.shape) == 2:
            relevance = tf.expand_dims(relevance, -1)
        relevance = tf.transpose(relevance, (1, 0, 2))
        if relevance.shape[-1] < self.hparams.dimension:
            raise ValueError('Vectorization requires the ranking model output size >= ' +
                             str(self.hparams.dimension) +
                             ", but get " + str(relevance.shape[-1])
                             + ". Please add 'output_size' in the hparams of this ranking model, "
                               "and adjust the size of build() method correspondingly.")
        relevance = relevance[:, :, :self.hparams.dimension]
        # (B, T, d)
        if self.hparams.affine == 1:
            relevance[:, :, 1:] = tf.ones_like(relevance[:, :, 1:])

        base_vector = self.get_base_vector_with_density(T)

        if not is_predict:
            propensity = self.build_propensity_model(relevance,
                                                     self.exp_settings.get(
                                                         'selection_bias_cutoff', 10),
                                                     self.hparams.dimension)
            # since observation model is initialized to ones,
            # we train relevance model first to ensure early stability.
            if self.hparams.affine == 0:
                propensity = tf.where(
                    tf.greater_equal(
                        self.global_step,
                        self.hparams.pretrain_ranker_step),
                    propensity,
                    tf.stop_gradient(propensity)
                )
        else:
            propensity = None
        return relevance, propensity, base_vector

    def build_propensity_model(self, relevance, T, dimension):
        if not hasattr(
                self, "_propensity_model") or self._propensity_model is None:
            initializer = tf.initializers.constant(1.0)
            self._propensity_model = tf.get_variable("pbm_weight", (1, T, dimension),
                                                     initializer=initializer)
        batch_size = tf.shape(relevance)[0]
        return tf.tile(self._propensity_model, [batch_size, 1, 1])  # (B, T, d)

    def build_observation_density_loss(self, propensities):
        # propensities: (B, T, d)
        # return: loss
        mean, log_var = self.build_observation_density_model(
            self.rank_list_size)  # (B, T, d)
        can_start_training = tf.greater_equal(
            self.global_step, self.hparams.pretrain_ranker_step)
        mean = tf.where(can_start_training, mean, tf.stop_gradient(mean))
        log_var = tf.where(
            can_start_training,
            log_var,
            tf.stop_gradient(log_var))
        self.scalar(log_var, "log_var", train_only=True)
        mean_loss = tf.reduce_mean(
            tf.squared_difference(mean, tf.stop_gradient(
                propensities)) * tf.exp(-log_var)
        )
        var_loss = tf.reduce_mean(log_var)
        l2_loss = 0
        for m in self._density_model:
            kernel = m.kernel
            l2_loss += tf.nn.l2_loss(kernel) * self.hparams.prob_l2_loss
            print(
                "Add density kernel L2 reg: " +
                str(kernel),
                self.hparams.prob_l2_loss)
        loss = mean_loss + var_loss + l2_loss
        self.scalar(mean_loss, "density_mean_loss", train_only=True)
        self.scalar(var_loss, "density_var_loss", train_only=True)
        self.scalar(l2_loss, "density_l2_loss", train_only=True)
        return loss

    def get_base_vector_with_density(self, T):
        mean, log_var = self.build_observation_density_model(T)  # (B, T, d)
        docid_inputs_tensor = tf.expand_dims(
            tf.transpose(tf.stack(self.docid_inputs[:T])), axis=-1)
        valid_flag = tf.where(
            tf.equal(
                docid_inputs_tensor,
                tf.cast(
                    tf.shape(
                        self.letor_features)[0],
                    tf.int64)),
            tf.zeros_like(docid_inputs_tensor, dtype=tf.float32),
            tf.ones_like(docid_inputs_tensor, dtype=tf.float32)
        )  # (B, T, 1)
        weight = tf.exp(-log_var) * valid_flag  # (B, T, d)
        base_vector = tf.reduce_mean(mean * weight, axis=1, keepdims=True) / \
            tf.reduce_mean(weight, axis=1, keepdims=True)  # (B, 1, d)
        return base_vector

    def build_observation_density_model(self, T):
        D = self.hparams.dimension
        features = self.get_input_feature_list(
            self.docid_inputs[:T])  # [T, (B, F)]
        features = tf.stop_gradient(
            tf.transpose(
                tf.stack(features), [
                    1, 0, 2]))  # (B, T, F)
        if not hasattr(self, "_density_model") or self._density_model is None:
            self._density_model = [
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(64, activation="elu"),
                tf.keras.layers.Dense(D * 2),
            ]
        x = features

        with tf.name_scope('density'):
            for m in self._density_model:
                x = m(x)
        mean = x[:, :, :D]
        log_var = x[:, :, D:]
        return mean, log_var

    def scalar(self, tensor, name, eval_only=False, train_only=False):
        collections = ['eval'] if eval_only else ['eval', 'train']
        if train_only:
            collections = ['train']
        tf.summary.scalar(
            name,
            tf.reduce_mean(tensor),
            collections=collections)

    def get_input_feature_list(self, input_id_list):
        """Copy from base_algorithm.get_ranking_scores()
        """
        PAD_embed = tf.zeros([1, self.feature_size], dtype=tf.float32)
        letor_features = tf.concat(
            axis=0, values=[
                self.letor_features, PAD_embed])
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                tf.nn.embedding_lookup(
                    letor_features, input_id_list[i]))
        return input_feature_list

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
            output_feed = [self.update,  # Update Op that does SGD.
                           self.loss,  # Loss for this batch.
                           self.train_summary  # Summarize statistics.
                           ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output  # Model outputs
            ]

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[2]
        else:
            return None, outputs[1], outputs[0]  # no loss, outputs, summary.
