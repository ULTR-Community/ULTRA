import os,sys
import json
import operator
import numpy as np
import math
FEATEURE_STATISTIC_FILE = sys.argv[1]
FEATURE_FILE = sys.argv[2]
output_file = sys.argv[3]
func=lambda x:x
# print(sys.argv)
if len(sys.argv)==5 and sys.argv[4]=="log":
#     print("using log")
    func=lambda x:math.log10(x+1)
#read feature scale
feature_scale = []
with open(FEATEURE_STATISTIC_FILE) as fin:
    feature_scale = json.load(fin)
feature_num = len(feature_scale)

def process(value,feature_scale_ind,func):
    ## Using log(1+x) to process, since many features are 0.
    ## this method is used to do with datasets like Istella which have many values around the min and max, and the max
    ## is way larger than min (10^50 larger). Naively using linear norm will make many small values not significant. 
    value=value-feature_scale_ind[0]
    rescale=[0,func(feature_scale_ind[1]-feature_scale_ind[0])]
#     print(rescale)
    scale=rescale[1] - rescale[0]
    if scale > 0:
        result = (func(value) -rescale[0])/ scale * 2 - 1
    else:
        result = func(value)
    return float("{0:.6g}".format(result))
#read feature: sort by qid, normalize and pad
with open(output_file,'w') as fout:
    def sort_data_by_qid(file_name):
        fin = open(file_name)
        data = []
        for line in fin:
            qid = int(line.split(' ')[1].split(':')[1])
            data.append((line, qid))
        fin.close()
        od = sorted(data, key=operator.itemgetter(1))
        return od
    od = sort_data_by_qid(FEATURE_FILE)
    for (line, qid) in od:
        arr = line.strip().split(' ')
        feature = np.zeros(feature_num)        
        fout.write(arr[0] + ' ' + arr[1])
        for i in range(len(arr)-2):
            arr2 = arr[i+2].split(':')
            idx = int(arr2[0])-1
            value = float(arr2[1])
            #TO DO: how to normalize? How to deal with empty value and the smallest value
            feature[idx]=process(value,feature_scale[idx],func)
        for i in range(feature_num):
            fout.write(' '+ str(i+1) + ':' + str(feature[i]))
        fout.write('\n')

