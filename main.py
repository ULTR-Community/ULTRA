"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.
    
    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18
    
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
import json

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import input_layer
import learning_algorithm

#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "./tmp_data/", "The directory of the experimental dataset.")
tf.app.flags.DEFINE_string("train_data_prefix", "train", "The name prefix of the training data in data_dir.")
tf.app.flags.DEFINE_string("valid_data_prefix", "valid", "The name prefix of the validation data in data_dir.")
tf.app.flags.DEFINE_string("test_data_prefix", "test", "The name prefix of the test data in data_dir.")
tf.app.flags.DEFINE_string("model_dir", "./tmp_model/", "The directory for model and intermediate outputs.")
tf.app.flags.DEFINE_string("output_dir", "./tmp_output/", "The directory to output results.")

# model 
tf.app.flags.DEFINE_string("setting_file", "./example/offline_setting/dla_exp_settings.json", "A json file that contains all the settings of the algorithm.")

# general training parameters
tf.app.flags.DEFINE_integer("batch_size", 256,
                            "Batch size to use during training.")
#tf.app.flags.DEFINE_integer("train_list_cutoff", 0,
#                            "The number of top documents to consider in each rank list during training (0: no limit).")
tf.app.flags.DEFINE_integer("max_list_cutoff", 0,
                            "The maximum number of top documents to consider in each rank list (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_iteration", 10000,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("start_saving_iteration", 0,
                            "The minimum number of iterations before starting to test and save models. (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("test_while_train", False,
                            "Set to True to test models during the training process.")
tf.app.flags.DEFINE_boolean("test_only", False,
                            "Set to True for testing models only.")


FLAGS = tf.app.flags.FLAGS

def create_model(session, exp_settings, data_set, forward_only):
    """Create model and initialize or load parameters in session.
    
        Args:
            session: (tf.Session) The session used to run tensorflow models
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
            forward_only: Set true to conduct prediction only, false to conduct training.
    """
    
    model = utils.find_class(exp_settings['learning_algorithm'])(data_set, exp_settings, forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train(exp_settings):
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)
    train_set = utils.read_data(FLAGS.data_dir, FLAGS.train_data_prefix, FLAGS.max_list_cutoff)
    utils.find_class(exp_settings['train_input_feed']).preprocess_data(train_set, exp_settings['train_input_hparams'], exp_settings)
    valid_set = utils.read_data(FLAGS.data_dir, FLAGS.valid_data_prefix, FLAGS.max_list_cutoff)
    utils.find_class(exp_settings['train_input_feed']).preprocess_data(valid_set, exp_settings['train_input_hparams'], exp_settings)
    print("Train Rank list size %d" % train_set.rank_list_size)
    print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)
    test_set = None
    if FLAGS.test_while_train:
        test_set = utils.read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
        utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
        print("Test Rank list size %d" % test_set.rank_list_size)
        exp_settings['max_candidate_num'] = max(test_set.rank_list_size, exp_settings['max_candidate_num'])
        test_set.pad(exp_settings['max_candidate_num'])

    if 'train_list_cutoff' not in exp_settings: # check if there is a limit on the number of items per training query.
        exp_settings['train_list_cutoff'] = exp_settings['max_candidate_num']
    else:
        exp_settings['train_list_cutoff'] = min(exp_settings['train_list_cutoff'], exp_settings['max_candidate_num'])
    
    # Pad data
    train_set.pad(exp_settings['max_candidate_num'])
    valid_set.pad(exp_settings['max_candidate_num'])
        

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model based on the input layer.
        print("Creating model...")
        model = create_model(sess, exp_settings, train_set, False)
        #model.print_info()

        # Create data feed
        train_input_feed = utils.find_class(exp_settings['train_input_feed'])(model, FLAGS.batch_size, exp_settings['train_input_hparams'], sess)
        valid_input_feed = utils.find_class(exp_settings['valid_input_feed'])(model, FLAGS.batch_size, exp_settings['valid_input_hparams'], sess)
        test_input_feed = None
        if FLAGS.test_while_train:
            test_input_feed = utils.find_class(exp_settings['test_input_feed'])(model, FLAGS.batch_size, exp_settings['test_input_hparams'], sess)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/train_log',
                                        sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.model_dir + '/valid_log')
        test_writer = None
        if FLAGS.test_while_train:
            test_writer = tf.summary.FileWriter(FLAGS.model_dir + '/test_log')

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        best_perf = None
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True)
            step_loss, _, summary = model.step(sess, input_feed, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            train_writer.add_summary(summary, model.global_step.eval())

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                print ("global step %d learning rate %.4f step-time %.2f loss "
                             "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                 step_time, loss))
                previous_losses.append(loss)
                # Validate model
                def validate_model(data_set, data_input_feed):
                    it = 0
                    count_batch = 0.0
                    summary_list = []
                    batch_size_list = []
                    while it < len(data_set.initial_list):
                        input_feed, info_map = data_input_feed.get_next_batch(it, data_set, check_validation=False)
                        _, _, summary = model.step(sess, input_feed, True)
                        summary_list.append(summary)
                        batch_size_list.append(len(info_map['input_list']))
                        it += batch_size_list[-1]
                        count_batch += 1.0
                    return utils.merge_TFSummary(summary_list, batch_size_list)
                    
                valid_summary = validate_model(valid_set, valid_input_feed)
                valid_writer.add_summary(valid_summary, model.global_step.eval())
                print("  valid: %s" % (
                    ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in valid_summary.value])
                ))

                if FLAGS.test_while_train:
                    test_summary = validate_model(test_set, test_input_feed)
                    test_writer.add_summary(test_summary, model.global_step.eval())
                    print("  test: %s" % (
                    ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])
                    ))

                # Save checkpoint if the objective metric on the validation set is better
                if "objective_metric" in exp_settings:
                    for x in valid_summary.value:
                        if x.tag == exp_settings["objective_metric"]:
                            if current_step >= FLAGS.start_saving_iteration:
                                if best_perf == None or best_perf < x.simple_value:
                                    checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                                    best_perf = x.simple_value
                                    print('Save model, valid %s:%.3f' % (x.tag, best_perf))
                                    break
                # Save checkpoint if there is no objective metic
                if best_perf == None and current_step > FLAGS.start_saving_iteration:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                if loss == float('inf'):
                    break

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

                if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                    break
                



def test(exp_settings):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = utils.read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
        utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
        exp_settings['max_candidate_num'] = test_set.rank_list_size

        test_set.pad(exp_settings['max_candidate_num'])
        

        # Create model and load parameters.
        model = create_model(sess, exp_settings, test_set, True)

        # Create input feed
        #test_input_feed = utils.find_class(exp_settings['test_input_feed'])(model, 1, exp_settings['test_input_hparams'], sess)
        test_input_feed = utils.find_class(exp_settings['test_input_feed'])(model, FLAGS.batch_size, exp_settings['test_input_hparams'], sess)

        test_writer = tf.summary.FileWriter(FLAGS.model_dir + '/test_log')

        rerank_scores = []
        summary_list = []
        # Start testing.
        
        it = 0
        count_batch = 0.0
        batch_size_list = []
        while it < len(test_set.initial_list):
            input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
            _, output_logits, summary = model.step(sess, input_feed, True)
            summary_list.append(summary)
            batch_size_list.append(len(info_map['input_list']))
            for x in range(batch_size_list[-1]):
                rerank_scores.append(output_logits[x])
            it += batch_size_list[-1]
            count_batch += 1.0
            print("Testing {:.0%} finished".format(float(it)/len(test_set.initial_list)), end="\r", flush=True)
            
        print("\n[Done]")
        test_summary = utils.merge_TFSummary(summary_list, batch_size_list)
        test_writer.add_summary(test_summary, it)
        print("  eval: %s" % (
                ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])
        ))

        #get rerank indexes with new scores
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = rerank_scores[i]
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        utils.output_ranklist(test_set, rerank_scores, FLAGS.output_dir, FLAGS.test_data_prefix)

    return

def main(_):
    exp_settings = json.load(open(FLAGS.setting_file))
    if FLAGS.test_only:
        test(exp_settings)
    else:
        train(exp_settings)

if __name__ == "__main__":
    tf.app.run()
