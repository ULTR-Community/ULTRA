import os
import sys
import json
import copy
import random
from ultra.utils import data_utils
from ultra.utils import click_models as CM


class BasicPropensityEstimator:
    def __init__(self, file_name=None):
        """Initialize a propensity estimator.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
                        'None' means creating from scratches.

        """
        if file_name:
            self.loadEstimatorFromFile(file_name)

    def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
        """Computing the propensity weights for each result in a list with clicks.

        Args:
            click_list: list<int> The list of clicks indicating whether each result are clicked (>0) or not (=0).
            use_non_clicked_data: Set True to give weights to non-clicked results, otherwise the non-clicked results
                                    would have 0 weights.

        Returns:
            propensity_weights: list<float> A list of propensity weights for the corresponding results.
        """
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data or (click_list[r] > 0):
                if r < len(self.IPW_list):
                    pw = self.IPW_list[r]
                else:
                    pw = self.IPW_list[-1]
            propensity_weights.append(pw)
        return propensity_weights

    def loadEstimatorFromFile(self, file_name):
        """Load a propensity estimator from a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        with open(file_name) as data_file:
            data = json.load(data_file)
            self.IPW_list = data['IPW_list']
        return

    def outputEstimatorToFile(self, file_name):
        """Export a propensity estimator to a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        json_dict = {
            'IPW_list': self.IPW_list
        }
        with open(file_name, 'w') as fout:
            fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
        return


class RandomizedPropensityEstimator(BasicPropensityEstimator):
    def __init__(self, file_name=None):
        """Initialize a propensity estimator.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
                        'None' means creating from scratches.

        """
        if file_name:
            self.loadEstimatorFromFile(file_name)

    def loadEstimatorFromFile(self, file_name):
        """Load a propensity estimator from a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        with open(file_name) as data_file:
            data = json.load(data_file)
            self.click_model = None
            if 'click_model' in data:
                self.click_model = CM.loadModelFromJson(data['click_model'])
            self.IPW_list = data['IPW_list']
        return

    def estimateParametersFromModel(self, click_model, data_set):
        """Estimate propensity weights based on clicks simulated with a click model.

        Args:
            click_model: (ClickModel) The click model used to generate clicks.
            data_set: (Raw_data) The data set with rank lists and labels.

        """
        self.click_model = click_model
        click_count = [[0 for _ in range(x + 1)]
                       for x in range(data_set.rank_list_size)]
        label_lists = copy.deepcopy(data_set.labels)
        # Conduct randomized click experiments
        session_num = 0
        while session_num < 10e6:
            index = random.randint(0, len(label_lists) - 1)
            random.shuffle(label_lists[index])
            click_list, _, _ = self.click_model.sampleClicksForOneList(
                label_lists[index])
            # Count how many clicks happened on the i position for a list with
            # that lengths.
            for i in range(len(click_list)):
                click_count[len(click_list) - 1][i] += click_list[i]
            session_num += 1
        # Count how many clicks happened on the 1st position for a list with
        # different lengths.
        first_click_count = [0 for _ in range(data_set.rank_list_size)]
        agg_click_count = [0 for _ in range(data_set.rank_list_size)]
        for x in range(len(click_count)):
            for y in range(x, len(click_count)):
                first_click_count[x] += click_count[y][0]
                agg_click_count[x] += click_count[y][x]

        # Estimate IPW for each position (assuming that position 0 has weight
        # 1)
        self.IPW_list = [min(first_click_count[x] / (agg_click_count[x] + 10e-6),
                             first_click_count[x]) for x in range(len(click_count))]
        return

    def outputEstimatorToFile(self, file_name):
        """Export a propensity estimator to a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        json_dict = {
            'click_model': self.click_model.getModelJson(),
            'IPW_list': self.IPW_list
        }
        with open(file_name, 'w') as fout:
            fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
        return


class OraclePropensityEstimator(BasicPropensityEstimator):

    def __init__(self, click_model):
        self.click_model = click_model

    def loadEstimatorFromFile(self, file_name):
        """Load a propensity estimator from a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        with open(file_name) as data_file:
            data = json.load(data_file)
            self.click_model = CM.loadModelFromJson(data['click_model'])
        return

    def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
        return self.click_model.estimatePropensityWeightsForOneList(
            click_list, use_non_clicked_data)

    def outputEstimatorToFile(self, file_name):
        """Export a propensity estimator to a json file.

        Args:
            file_name: (string) The path to the json file of the propensity estimator.
        """
        json_dict = {
            'click_model': self.click_model.getModelJson(),
        }
        with open(file_name, 'w') as fout:
            fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
        return


def main():
    click_model_json_file = sys.argv[1]
    data_dir = sys.argv[2]
    output_path = sys.argv[3]

    print("Load data from " + data_dir)
    train_set = data_utils.read_data(data_dir, 'train')
    click_model = None
    with open(click_model_json_file) as fin:
        model_desc = json.load(fin)
        click_model = CM.loadModelFromJson(model_desc)
    print("Estimating...")
    estimator = RandomizedPropensityEstimator()
    estimator.estimateParametersFromModel(click_model, train_set)
    print("Output results...")
    output_file = output_path + '/randomized_' + \
        click_model_json_file.split('/')[-1][:-5] + '.json'
    estimator.outputEstimatorToFile(output_file)


if __name__ == "__main__":
    main()
