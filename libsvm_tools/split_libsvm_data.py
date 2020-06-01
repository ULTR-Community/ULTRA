import os, sys
import random

input_file = sys.argv[1]
valid_file = sys.argv[2]
train_file = sys.argv[3]
sample_rate = float(sys.argv[4])
qid_list_data = {}
with open(input_file) as fin:
    for line in fin:
        arr = line.strip().split(' ')
        qid = int(arr[1].split(':')[1])
        if qid not in qid_list_data:
            qid_list_data[qid] = []
        qid_list_data[qid].append(line.strip())

qid_list_data=list(qid_list_data.items())
random.shuffle(qid_list_data)
ind_valid=max(int(sample_rate * len(qid_list_data)), 1)
valid_qid_lists=qid_list_data[:ind_valid]
train_qid_lists=qid_list_data[ind_valid:]

with open(valid_file, 'w') as fout:
    sorted_qid_lists = sorted(valid_qid_lists, key=lambda k: k[0])
    for qid_list in sorted_qid_lists:
        for line in qid_list[1]:
            fout.write(line)
            fout.write('\n')
with open(train_file, 'w') as fout:
    sorted_qid_lists = sorted(train_qid_lists, key=lambda k: k[0])
    for qid_list in sorted_qid_lists:
        for line in qid_list[1]:
            fout.write(line)
            fout.write('\n')


