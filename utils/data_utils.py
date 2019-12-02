# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import json
import random
import os
from . import metrics
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

class Raw_data:
    def __init__(self, data_path = None, file_prefix = None, rank_cut=None):
        """
        Initialize a dataset

        Args:
            data_path: (string) the root directory of the experimental dataset.
            file_prefix: (string) the prefix of the data to process, e.g. 'train', 'valid', or 'test'.
            rank_cut: (int) the maximum number of top documents considered in each list.

        Returns:
            None
        """
        if data_path == None:
            self.feature_size = -1
            self.rank_list_size = -1
            self.features = []
            self.dids = []
            self.initial_list = []
            self.qids = []
            return

        settings = json.load(open(data_path + 'settings.json'))
        self.feature_size = settings['feature_size']
        metrics.RankingMetricKey.MAX_LABEL = settings['max_label']

        self.features = []
        self.dids = []
        feature_fin = open(data_path + file_prefix + '/' + file_prefix + '.feature')
        for line in feature_fin:
            arr = line.strip().split(' ')
            self.dids.append(arr[0])
            self.features.append(np.zeros(self.feature_size))
            for x in arr[1:]:
                arr2 = x.split(':')
                self.features[-1][int(arr2[0])] = float(arr2[1])
        feature_fin.close()

        self.rank_list_size = 0
        self.initial_list = []
        self.qids = []
        init_list_fin = open(data_path + file_prefix + '/' + file_prefix + '.init_list')
        for line in init_list_fin:
            arr = line.strip().split(' ')
            self.qids.append(arr[0])
            if rank_cut:
                self.initial_list.append([int(x) for x in arr[1:][:rank_cut]])
            else:
                self.initial_list.append([int(x) for x in arr[1:]])
            if len(self.initial_list[-1]) > self.rank_list_size:
                self.rank_list_size = len(self.initial_list[-1])
        init_list_fin.close()

        self.labels = []
        label_fin = open(data_path + file_prefix + '/' + file_prefix + '.labels')
        for line in label_fin:
            self.labels.append([float(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])
        label_fin.close()

        self.initial_scores = []
        if os.path.isfile(data_path + file_prefix + '/' + file_prefix + '.intial_scores'):
            with open(data_path + file_prefix + '/' + file_prefix + '.initial_scores') as fin:
                for line in fin:
                    self.initial_scores.append([float(x) for x in line.strip().split(' ')[1:]])
        
        self.initial_list_lengths = [len(self.initial_list[i]) for i in range(len(self.initial_list))]

    def pad(self, rank_list_size, pad_tails = True):
        """
        Pad a rank list with zero feature vectors when it is shorter than the required rank list size.

        Args:
            rank_list_size: (int) the required size of a ranked list
            pad_tails: (bool) Add padding vectors to the tails of each list (True) or the heads of each list (False) 

        Returns:
            None
        """
        self.rank_list_size = rank_list_size
        self.features.append(np.zeros(self.feature_size))  # vector for pad

        for i in range(len(self.initial_list)):
            if len(self.initial_list[i]) < self.rank_list_size:
                if pad_tails: # pad tails
                    self.initial_list[i] += [-1] * (self.rank_list_size - len(self.initial_list[i]))
                else:    # pad heads
                    self.initial_list[i] = [-1] * (self.rank_list_size - len(self.initial_list[i])) + self.initial_list[i]

def merge_TFSummary(summary_list, weights):
    merged_values = {}
    weight_sum_map = {}
    for i in range(len(summary_list)):
        summary = summary_list[i]
        if isinstance(summary, bytes):
            parse_TFSummary_from_bytes(summary)
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        for e in summary.value:
            if e.tag not in merged_values:
                merged_values[e.tag] = 0.0
                weight_sum_map[e.tag] = 0.0
            merged_values[e.tag] += e.simple_value * weights[i]
            weight_sum_map[e.tag] += weights[i]
    for k in merged_values:
        merged_values[k] /= max(0.0000001, weight_sum_map[k])
    return tf.Summary(value=[ 
                tf.Summary.Value(tag=k, simple_value=merged_values[k]) for k in merged_values  
            ]) 

def parse_TFSummary_from_bytes(summary_bytes):
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_bytes)
    return {x.tag:x.simple_value for x in summary.value}

def read_data(data_path, file_prefix, rank_cut = None):
    data = Raw_data(data_path, file_prefix, rank_cut)
    return data

def generate_ranklist(data, rerank_lists):
    """
        Create a reranked lists based on the data and rerank documents ids.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_lists: (list<list<int>>) a list of rerank list in which each 
                            element represents the original rank of the documents 
                            in the initial list.

        Returns:
            qid_list_map: (map<list<int>>) a map of qid with the reranked document id list.
    """
    if len(rerank_lists) != len(data.initial_list):
        raise ValueError("The number of queries in rerank ranklists number must be equal to the initial list,"
                         " %d != %d." % (len(rerank_lists)), len(data.initial_list))
    qid_list_map = {}
    for i in range(len(data.qids)):
        if len(rerank_lists[i]) != len(data.initial_list[i]):
            raise ValueError("The number of docs in each rerank ranklists must be equal to the initial list,"
                             " %d != %d." % (len(rerank_lists[i]), len(data.initial_list[i])))
        #remove duplicated docs and organize rerank list
        index_list = []
        index_set = set()
        for idx in rerank_lists[i]:
            if idx not in index_set:
                index_set.add(idx)
                index_list.append(idx)
        # doc idxs that haven't been observed in the rerank list will be put at the end of the list
        for idx in range(len(rerank_lists[i])): 
            if idx not in index_set:
                index_list.append(idx)
        #get new ranking list
        qid = data.qids[i]
        did_list = []
        new_list = [data.initial_list[i][idx] for idx in index_list]
        # remove padding documents
        for ni in new_list:
            if ni >= 0:
                did_list.append(data.dids[ni])
        qid_list_map[qid] = did_list
    return qid_list_map

def generate_ranklist_by_scores(data, rerank_scores):
    """
        Create a reranked lists based on the data and rerank scores.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_scores: (list<list<float>>) a list of rerank list in which each 
                            element represents the reranking scores for the documents 
                            on that position in the initial list.

        Returns:
            qid_list_map: (map<list<int>>) a map of qid with the reranked document id list.
    """
    if len(rerank_scores) != len(data.initial_list):
        raise ValueError("Rerank ranklists number must be equal to the initial list,"
                         " %d != %d." % (len(rerank_scores)), len(data.initial_list))
    qid_list_map = {}
    for i in range(len(data.qids)):
        scores = rerank_scores[i]
        rerank_list = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        if len(rerank_list) != len(data.initial_list[i]):
            raise ValueError("Rerank ranklists length must be equal to the gold list,"
                             " %d != %d." % (len(rerank_scores[i]), len(data.initial_list[i])))
        #remove duplicate and organize rerank list
        index_list = []
        index_set = set()
        for idx in rerank_list:
            if idx not in index_set:
                index_set.add(idx)
                index_list.append(idx)
        # doc idxs that haven't been observed in the rerank list will be put at the end of the list
        for idx in range(len(rerank_list)):
            if idx not in index_set:
                index_list.append(idx)
        #get new ranking list
        qid = data.qids[i]
        did_list = []
        # remove padding documents
        for idx in index_list:
            ni = data.initial_list[i][idx]
            ns = scores[idx]
            if ni >= 0:
                did_list.append((data.dids[ni], ns))
        qid_list_map[qid] = did_list
    return qid_list_map

def output_ranklist(data, rerank_scores, output_path, file_name = 'test'):
    """
        Create a trec format rank list by reranking the initial list with reranking scores.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_scores: (list<list<float>>) a list of rerank list in which each 
                            element represents the reranking scores for the documents 
                            on that position in the initial list.
            output_path: (string) the path for the output
            file_name: (string) the name of the output set, e.g., 'train', 'valid', 'text'.

        Returns:
            None
    """
    qid_list_map = generate_ranklist_by_scores(data, rerank_scores)
    fout = open(output_path + file_name + '.ranklist','w')
    for qid in data.qids:
        for i in range(len(qid_list_map[qid])):
            fout.write(qid + ' Q0 ' + qid_list_map[qid][i][0] + ' ' + str(i+1)
                            + ' ' + str(qid_list_map[qid][i][1]) + ' Model\n')
    fout.close()

