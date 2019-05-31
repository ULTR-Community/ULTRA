import os,sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(output_file, 'w') as fout:
    with open(input_file) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            feature_list = arr[2:]
            idx_feature_map = {int(x.split(':')[0]) : x for x in feature_list}
            sorted_idx_feature_list = sorted(idx_feature_map.items(), key=lambda k: k[0])
            fout.write(' '.join([arr[0], arr[1]] + [x[1] for x in sorted_idx_feature_list]))
            fout.write('\n')