"""Test basic class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
from parameterized import parameterized

test_setting_list = []
for f in os.listdir('./tests/test_settings/'):
    if f.endswith('.json'):
        test_setting_list.append(f[:-5])


class TestBasicModel(unittest.TestCase):
    """Test model pipeline."""

    @parameterized.expand(test_setting_list)
    def test_building(self, json_file_name):
        """Test model building."""
        train_command = ' '.join([
            'python main.py',
            '--max_train_iteration=20',
            '--setting_file=./tests/test_settings/%s.json' % json_file_name,
            '--model_dir=./tests/tmp_model_%s/' % json_file_name,
            '--output_dir=./tests/tmp_output_%s/' % json_file_name,
        ])
        os.system(train_command)

        meta_saved = False
        index_saved = False
        for f in os.listdir('./tests/tmp_model_%s/' % json_file_name):
            if f.endswith('.meta'):
                meta_saved = True
            if f.endswith('.index'):
                index_saved = True

        print('\n\n------------------------------')
        print('Build Test for %s.json' % json_file_name)
        print('------------------------------')

        print('Model meta saved? %s' % str(meta_saved))
        assert meta_saved

        print('Model index saved? %s' % str(index_saved))
        assert index_saved

    @parameterized.expand(test_setting_list)
    def test_prediction(self, json_file_name):
        """Test model prediction."""
        test_command = ' '.join([
            'python main.py',
            '--test_only=True',
            '--setting_file=./tests/test_settings/%s.json' % json_file_name,
            '--model_dir=./tests/tmp_model_%s/' % json_file_name,
            '--output_dir=./tests/tmp_output_%s/' % json_file_name,
        ])
        os.system(test_command)

        ranklist_saved = False
        for f in os.listdir('./tests/tmp_output_%s/' % json_file_name,):
            if f.endswith('.ranklist'):
                ranklist_saved = True

        print('\n\n------------------------------')
        print('Prediction Test for %s.json' % json_file_name)
        print('------------------------------')

        print('Ranked list saved? %s' % str(ranklist_saved))
        assert ranklist_saved

        os.system('rm -r ./tests/tmp_model_%s/' % json_file_name,)
        os.system('rm -r ./tests/tmp_output_%s/' % json_file_name,)
