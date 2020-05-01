import os
import sys
import random
import json


def loadModelFromJson(model_desc):
    click_model = PositionBiasedModel()
    if model_desc['model_name'] == 'user_browsing_model':
        click_model = UserBrowsingModel()
    elif model_desc['model_name'] == 'cascade_model':
        click_model = CascadeModel()
    click_model.eta = model_desc['eta']
    click_model.click_prob = model_desc['click_prob']
    click_model.exam_prob = model_desc['exam_prob']
    return click_model


class ClickModel:
    def __init__(self, neg_click_prob=0.0, pos_click_prob=1.0,
                 relevance_grading_num=1, eta=1.0):
        self.exam_prob = None
        self.setExamProb(eta)
        self.setClickProb(
            neg_click_prob,
            pos_click_prob,
            relevance_grading_num)

    @property
    def model_name(self):
        return 'click_model'

    # Serialize model into a json.
    def getModelJson(self):
        desc = {
            'model_name': self.model_name,
            'eta': self.eta,
            'click_prob': self.click_prob,
            'exam_prob': self.exam_prob
        }
        return desc

    # Generate noisy click probability based on relevance grading number
    # Inspired by ERR
    def setClickProb(self, neg_click_prob, pos_click_prob,
                     relevance_grading_num):
        b = (pos_click_prob - neg_click_prob) / \
            (pow(2, relevance_grading_num) - 1)
        a = neg_click_prob - b
        self.click_prob = [
            a + pow(2, i) * b for i in range(relevance_grading_num + 1)]

    # Set the examination probability for the click model.
    def setExamProb(self, eta):
        self.eta = eta
        return

    # Sample clicks for a list
    def sampleClicksForOneList(self, label_list):
        return None

    # Estimate propensity for clicks in a list
    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        return None


class PositionBiasedModel(ClickModel):

    @property
    def model_name(self):
        return 'position_biased_model'

    def setExamProb(self, eta):
        self.eta = eta
        self.original_exam_prob = [0.68, 0.61, 0.48,
                                   0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]
        self.exam_prob = [pow(x, eta) for x in self.original_exam_prob]

    def sampleClicksForOneList(self, label_list):
        click_list, exam_p_list, click_p_list = [], [], []
        for rank in range(len(label_list)):
            click, exam_p, click_p = self.sampleClick(rank, label_list[rank])
            click_list.append(click)
            exam_p_list.append(exam_p)
            click_p_list.append(click_p)
        return click_list, exam_p_list, click_p_list

    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data | click_list[r] > 0:
                pw = 1.0 / self.getExamProb(r) * self.getExamProb(0)
            propensity_weights.append(pw)
        return propensity_weights

    def sampleClick(self, rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        exam_p = self.getExamProb(rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        click = 1 if random.random() < exam_p * click_p else 0
        return click, exam_p, click_p

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]


class UserBrowsingModel(ClickModel):

    @property
    def model_name(self):
        return 'user_browsing_model'

    def setExamProb(self, eta):
        self.eta = eta
        self.original_rd_exam_table = [
            [1.0],
            [0.98, 1.0],
            [1.0, 0.62, 0.95],
            [1.0, 0.77, 0.42, 0.82],
            [1.0, 0.92, 0.55, 0.31, 0.69],
            [1.0, 0.96, 0.63, 0.4, 0.22, 0.54],
            [1.0, 0.99, 0.73, 0.46, 0.29, 0.17, 0.47],
            [1.0, 1.0, 0.89, 0.52, 0.35, 0.24, 0.14, 0.43],
            [1.0, 1.0, 0.95, 0.68, 0.4, 0.29, 0.19, 0.12, 0.41],
            [1.0, 1.0, 1.0, 0.96, 0.52, 0.36, 0.27, 0.18, 0.12, 0.43]
        ]
        self.exam_prob = []
        for i in range(len(self.original_rd_exam_table)):
            self.exam_prob.append([pow(x, eta)
                                   for x in self.original_rd_exam_table[i]])

    def sampleClicksForOneList(self, label_list):
        click_list, exam_p_list, click_p_list = [], [], []
        last_click_rank = -1
        for rank in range(len(label_list)):
            click, exam_p, click_p = self.sampleClick(
                rank, last_click_rank, label_list[rank])
            if click > 0:
                last_click_rank = rank
            click_list.append(click)
            exam_p_list.append(exam_p)
            click_p_list.append(click_p)
        return click_list, exam_p_list, click_p_list

    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        last_click_rank = -1
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data | click_list[r] > 0:
                pw = 1.0 / self.getExamProb(r, last_click_rank)
            if click_list[r] > 0:
                last_click_rank = r
            propensity_weights.append(pw)
        return propensity_weights

    def sampleClick(self, rank, last_click_rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        exam_p = self.getExamProb(rank, last_click_rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        click = 1 if random.random() < exam_p * click_p else 0
        return click, exam_p, click_p

    def getExamProb(self, rank, last_click_rank):
        distance = rank - last_click_rank
        if rank < len(self.exam_prob):
            exam_p = self.exam_prob[rank][distance - 1]
        else:
            if distance > rank:
                exam_p = self.exam_prob[-1][-1]
            else:
                idx = distance - \
                    1 if distance < len(self.exam_prob[-1]) - 1 else -2
                exam_p = self.exam_prob[-1][idx]
        return exam_p


class CascadeModel(ClickModel):

    @property
    def model_name(self):
        return 'cascade_model'

    def setExamProb(self, eta):
        self.eta = eta
        self.exam_prob = [1.0 for _ in range(10)]

    def sampleClicksForOneList(self, label_list):
        click_list, exam_p_list, click_p_list = [], [], []
        has_click = False
        for rank in range(len(label_list)):
            click, exam_p, click_p = self.sampleClick(rank, label_list[rank])
            if has_click:
                click_list.append(0.0)
                exam_p_list.append(0.0)
            else:
                click_list.append(click)
                exam_p_list.append(exam_p)
            click_p_list.append(click_p)
            if click > 0:
                has_click = True
        return click_list, exam_p_list, click_p_list

    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data | click_list[r] > 0:
                pw = 1.0 / self.getExamProb(r) * self.getExamProb(0)
            propensity_weights.append(pw)
        return propensity_weights

    def sampleClick(self, rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        exam_p = self.getExamProb(rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        click = 1 if random.random() < exam_p * click_p else 0
        return click, exam_p, click_p

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]


def test_initialization():
    # Test PBM
    test_model = PositionBiasedModel(0.1, 0.9, 4, 1.0)
    print('PBM(3, 4) -> %d, %f, %f' % test_model.sampleClick(3, 4))
    print('PBM(2, 0) -> %d, %f, %f' % test_model.sampleClick(2, 0))
    print('PBM(14, 1) -> %d, %f, %f' % test_model.sampleClick(14, 1))
    click_list, exam_p_list, click_p_list = test_model.sampleClicksForOneList([
                                                                              4, 0, 3, 4])
    print(click_list)
    print(exam_p_list)
    print(click_p_list)
    print(test_model.estimatePropensityWeightsForOneList(click_list))

    # Test UBM
    test_model = UserBrowsingModel(0.1, 0.9, 4, 1.0)
    print('UBM(3, 0, 4) -> %d, %f, %f' % test_model.sampleClick(3, 0, 4))
    print('UBM(14, -1, 0) -> %d, %f, %f' % test_model.sampleClick(14, -1, 0))
    print('UBM(14, 9, 1) -> %d, %f, %f' % test_model.sampleClick(14, 9, 1))
    print('UBM(14, 1, 2) -> %d, %f, %f' % test_model.sampleClick(14, 1, 2))
    click_list, exam_p_list, click_p_list = test_model.sampleClicksForOneList([
                                                                              4, 0, 3, 4])
    print(click_list)
    print(exam_p_list)
    print(click_p_list)
    print(test_model.estimatePropensityWeightsForOneList(click_list))


def test_load_from_file():
    file_name = sys.argv[1]
    click_model = None
    with open(file_name) as fin:
        data = json.load(fin)
        click_model = loadModelFromJson(data)
    click_list, exam_p_list, click_p_list = click_model.sampleClicksForOneList([
                                                                               4, 0, 3, 4])
    print(click_list)
    print(exam_p_list)
    print(click_p_list)
    print(click_model.estimatePropensityWeightsForOneList(click_list))


def main():
    MODELS = {
        'pbm': PositionBiasedModel,
        'cascade': CascadeModel,
        'ubm': UserBrowsingModel,
    }

    model_name = sys.argv[1]
    neg_click_prob = float(sys.argv[2])
    pos_click_prob = float(sys.argv[3])
    max_relevance_grade = int(sys.argv[4])
    eta = float(sys.argv[5])
    output_path = sys.argv[6]

    click_model = MODELS[model_name](neg_click_prob, pos_click_prob,
                                     max_relevance_grade, eta)

    with open(output_path + '/' + '_'.join(sys.argv[1:6]) + '.json', 'w') as fout:
        fout.write(
            json.dumps(
                click_model.getModelJson(),
                indent=4,
                sort_keys=True))


if __name__ == "__main__":
    # test_load_from_file()
    main()
