import os
import sys
import random
import math
import json

list_lengths = []
max_label = 0


def read_one_instance(feature_fin, rank_score_fin):
    feature_line = feature_fin.readline()
    score_line = rank_score_fin.readline()
    if feature_line == '' or score_line == '':
        return None, None, None, None

    arr = feature_line.strip().split(' ')
    if len(arr) < 3:
        print('thing wrong')
    label = float(arr[0])
    qid = arr[1].split(':')[1]
    features = [i for i in arr[2:]]
    score = float(score_line)
    return qid, features, label, score


def prepare_one_set(feature_path, rank_score_path, output_path, set_name):
    global max_label
    # read raw data and build data map
    feature_fin = open(feature_path + set_name + '.txt')
    rank_score_fin = open(rank_score_path + set_name + '.predict')

    qid_list = []
    qid_did_map, qid_feature_map, qid_label_map, qid_score_map = {}, {}, {}, {}
    qid, feature, label, score = read_one_instance(feature_fin, rank_score_fin)
    line_num = 0
    while qid is not None:
        if qid not in qid_did_map:
            qid_list.append(qid)
            qid_did_map[qid], qid_feature_map[qid], qid_label_map[qid], qid_score_map[qid] = [
            ], [], [], []
        did = set_name + '_' + qid + '_' + str(line_num)
        qid_did_map[qid].append(did)
        qid_feature_map[qid].append(feature)
        qid_label_map[qid].append(label)
        qid_score_map[qid].append(score)
        max_label = max(max_label, label)
        qid, feature, label, score = read_one_instance(
            feature_fin, rank_score_fin)
        line_num += 1
    feature_fin.close()
    rank_score_fin.close()

    # generate rank lists
    qid_initial_rank_map = {}
    for qid in qid_list:
        scores = qid_score_map[qid]
        rank_length = len(scores)
        list_lengths.append(rank_length)
        # qid_initial_rank_map[qid] store the indexes to raw data
        qid_initial_rank_map[qid] = sorted(
            range(
                len(scores)),
            key=lambda k: scores[k],
            reverse=True)[
            :rank_length]

    # output trec format rank list for offline evaluation
    qrel_fout = open(output_path + set_name + '.qrels', 'w')
    initial_trec_fout = open(output_path + set_name + '.trec.init_list', 'w')
    for qid in qid_list:
        for i in range(len(qid_initial_rank_map[qid])):
            idx = qid_initial_rank_map[qid][i]
            initial_trec_fout.write(qid + ' Q0 ' + qid_did_map[qid][idx] + ' ' + str(i + 1)
                                    + ' ' + str(qid_score_map[qid][idx]) + ' InitialRank\n')
        # output qrels
        for i in range(len(qid_did_map[qid])):
            qrel_fout.write(qid + ' 0 ' + qid_did_map[qid][i] + ' '
                            + str(int(qid_label_map[qid][i])) + '\n')
    qrel_fout.close()
    initial_trec_fout.close()

    # output experiment data
    feature_fout = open(output_path + set_name + '.feature', 'w')
    initial_rank_fout = open(output_path + set_name + '.init_list', 'w')
    label_fout = open(output_path + set_name + '.labels', 'w')
    initial_score_fout = open(output_path + set_name + '.initial_scores', 'w')
    line_num = 0  # a variable used to record the line number of the feature vector for each doc
    for qid in qid_list:
        initial_rank_fout.write(qid)
        label_fout.write(qid)
        initial_score_fout.write(qid)
        for i in range(len(qid_initial_rank_map[qid])):
            # output the corresponding line number of the features for the doc
            initial_rank_fout.write(' ' + str(line_num))
            idx = qid_initial_rank_map[qid][i]
            # the label of each doc
            label_fout.write(' ' + str(qid_label_map[qid][idx]))
            # the initial ranking scores from the initial model
            initial_score_fout.write(' ' + str(qid_score_map[qid][idx]))
            feature_fout.write(qid_did_map[qid][idx])
            for x in qid_feature_map[qid][idx]:
                # svmlight format feature index starts from 1, but we need it
                # to start from 0
                arr = x.split(':')
                feature_fout.write(' ' + str(int(arr[0]) - 1) + ':' + arr[1])
            feature_fout.write('\n')
            line_num += 1

        initial_rank_fout.write('\n')
        label_fout.write('\n')
        initial_score_fout.write('\n')
    initial_rank_fout.close()
    feature_fout.close()
    label_fout.close()
    initial_score_fout.close()


def main():
    global max_label
    DATA_PATH = sys.argv[1] + '/'
    INITIAL_RANK_PATH = sys.argv[2]  # The path of the SVMrank outputs
    OUTPUT_PATH = sys.argv[3]
    FEATURE_SIZE = int(sys.argv[4])  # The number of letor features in total
    SET_NAME = ['train', 'test', 'valid']

    for set_name in SET_NAME:
        if not os.path.exists(OUTPUT_PATH + set_name + '/'):
            os.makedirs(OUTPUT_PATH + set_name + '/')
        prepare_one_set(
            DATA_PATH,
            INITIAL_RANK_PATH,
            OUTPUT_PATH +
            set_name +
            '/',
            set_name)

    settings = {}
    settings['feature_size'] = FEATURE_SIZE
    settings['max_label'] = max_label
    set_fout = open(OUTPUT_PATH + 'settings.json', 'w')
    json.dump(settings, set_fout)
    set_fout.close()

    print('Longest list length %d' % (max(list_lengths)))
    print('Average list length %d' %
          (sum(list_lengths) / float(len(list_lengths))))


if __name__ == "__main__":
    main()
