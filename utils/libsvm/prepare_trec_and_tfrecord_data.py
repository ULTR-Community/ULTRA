import os,sys
import random
import numpy as np
import math
import json
import tensorflow as tf

list_lengths = []

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

def prepare_one_set(rank_cutoff, feature_path, rank_score_path, output_path ,set_name, feature_dim):
    #read raw data
    feature_fin = open(feature_path + set_name + '.txt')
    rank_score_fin = open(rank_score_path + set_name + '.predict')

    qid_list = []
    qid_did_map, qid_feature_map, qid_label_map, qid_score_map = {}, {}, {}, {}
    qid, feature, label, score = read_one_instance(feature_fin, rank_score_fin)
    line_num = 0
    while qid != None:
        if qid not in qid_did_map:
            qid_list.append(qid)
            qid_did_map[qid], qid_feature_map[qid], qid_label_map[qid], qid_score_map[qid] = [], [], [], []
        did = set_name + '_' + qid + '_' + str(line_num)
        qid_did_map[qid].append(did)
        qid_feature_map[qid].append(feature)
        qid_label_map[qid].append(label)
        qid_score_map[qid].append(score)
        qid, feature, label, score = read_one_instance(feature_fin, rank_score_fin)
        line_num += 1
    feature_fin.close()
    rank_score_fin.close()

    #generate rank lists with rank cutoff
    qid_initial_rank_map, qid_gold_rank_map = {}, {}
    for qid in qid_list:
        scores = qid_score_map[qid]
        rank_length = rank_cutoff if rank_cutoff < len(scores) else len(scores)
        list_lengths.append(rank_length)
        #qid_initial_rank_map[qid] store the indexes to raw data
        qid_initial_rank_map[qid] = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:rank_length]
        labels = [qid_label_map[qid][idx] for idx in qid_initial_rank_map[qid]]
        #qid_gold_rank_map[qid] store the rerank indexes to qid_initial_rank_map[qid]
        qid_gold_rank_map[qid] = sorted(range(len(labels)), key=lambda k: labels[k], reverse=True)

    #output evaluation rank list
    qrel_fout = open(output_path + set_name + '.qrels','w')
    initial_trec_fout = open(output_path + set_name + '.trec.init_list','w')
    gold_trec_fout = open(output_path + set_name + '.trec.gold_list','w')
    for qid in qid_list:
        for i in range(len(qid_initial_rank_map[qid])):
            idx = qid_initial_rank_map[qid][i]
            #qrel_fout.write(qid + ' 0 ' + qid_did_map[qid][idx] + ' ' 
            #                + str(int(qid_label_map[qid][idx])) + '\n')
            initial_trec_fout.write(qid + ' Q0 ' + qid_did_map[qid][idx] + ' ' + str(i+1)
                            + ' ' + str(qid_score_map[qid][idx]) + ' RankSVM\n')
            gold_idx = qid_initial_rank_map[qid][qid_gold_rank_map[qid][i]]
            gold_trec_fout.write(qid + ' Q0 ' + qid_did_map[qid][gold_idx] + ' ' + str(i+1)
                            + ' ' + str(qid_label_map[qid][gold_idx]) + ' Gold\n')
        #output qrels
        for i in range(len(qid_did_map[qid])):
            qrel_fout.write(qid + ' 0 ' + qid_did_map[qid][i] + ' ' 
                            + str(int(qid_label_map[qid][i])) + '\n')
    qrel_fout.close()
    initial_trec_fout.close()
    gold_trec_fout.close()

    #output TFRecord-based data
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.FeatureList(float_list=tf.train.FloatList(value=[value]))
        
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    line_num = 0
    tfrecord_fout = tf.data.experimental.TFRecordWriter(output_path + set_name + '.tfrecord')
    for qid in qid_list:
        # get feature list and label list
        sorted_feature_list = []
        label_list = []
        for i in range(len(qid_initial_rank_map[qid])):
            idx = qid_initial_rank_map[qid][i]
            doc_feature = np.zeros([feature_dim])
            for x in qid_feature_map[qid][idx]:
                arr = x.split(':')
                doc_feature[int(arr[0])-1] = float(arr[1])
            sorted_feature_list.append(doc_feature)
            label_list.append(qid_label_map[qid][idx])
        # create TF example
        example_proto = tf.train.Example(features=tf.train.Features(feature={
            'feature_list' : _float_feature(sorted_feature_list),
            'label_list' : _int64_feature(label_list) 
        }))
        tfrecord_fout.write(example_proto)

def main():
    DATA_PATH = sys.argv[1] + '/'
    INITIAL_RANK_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]
    RANK_CUT = int(sys.argv[4])
    FEATURE_DIM = int(sys.argv[5]) #700
    SET_NAME = ['train','test','valid']
    #SET_NAME = ['valid']


    for set_name in SET_NAME:
        if not os.path.exists(OUTPUT_PATH + set_name + '/'):
            os.makedirs(OUTPUT_PATH + set_name + '/')
        prepare_one_set(RANK_CUT, DATA_PATH, INITIAL_RANK_PATH, OUTPUT_PATH + set_name + '/', set_name, FEATURE_DIM)

    settings = {}
    settings['embed_size'] = FEATURE_DIM
    settings['rank_cutoff'] = RANK_CUT
    #settings['GO_embed'] = [random.random()/math.sqrt(float(FEATURE_DIM)) for _ in range(FEATURE_DIM)]
    set_fout = open(OUTPUT_PATH + 'settings.json','w')
    json.dump(settings, set_fout)
    set_fout.close()

    print('Longest list length %d' % (max(list_lengths)))
    print('Average list length %d' % (sum(list_lengths) / float(len(list_lengths))))

if __name__ == "__main__":
    main()


