"""Basic API needed for the implementation of an unbiased learning to rank algorithm.
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class BasicAlgorithm:
    """The basic class that contains all the API needed for the 
        implementation of an unbiased learning to rank algorithm.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.
    
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build BasicAlgorithm')

        self.hparams = tf.contrib.training.HParams()
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())
        self.updates = None
        self.loss = None
        self.output = None

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
            output_feed = [self.updates,    # Update Op that does SGD.
                            self.loss,    # Loss for this batch.
                            self.train_summary # Summarize statistics.
                            ]    
        else:
            output_feed = [self.loss, # Loss for this batch.
                        self.eval_summary, # Summarize statistics.
                        self.output   # Model outputs
            ]    

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], None, outputs[-1]    # loss, no outputs, summary.
        else:
            return outputs[0], outputs[2], outputs[1]    # loss, outputs, summary.
