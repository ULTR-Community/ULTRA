import os
import sys
import random

input_file = sys.argv[1]
output_file = sys.argv[2]
sample_rate = float(sys.argv[3])

qid_list_data = {}
with open(input_file) as fin:
    for line in fin:
        arr = line.strip().split(' ')
        qid = int(arr[1].split(':')[1])
        if qid not in qid_list_data:
            qid_list_data[qid] = []
        qid_list_data[qid].append(line.strip())

sampled_qid_lists = random.sample(qid_list_data.items(), max(
    int(sample_rate * len(qid_list_data)), 1))

with open(output_file, 'w') as fout:
    sorted_qid_lists = sorted(sampled_qid_lists, key=lambda k: k[0])
    for qid_list in sorted_qid_lists:
        for line in qid_list[1]:
            fout.write(line)
            fout.write('\n')
