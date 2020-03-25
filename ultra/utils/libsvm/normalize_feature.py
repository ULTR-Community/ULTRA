import os,sys
import json
import operator
import numpy as np

FEATEURE_STATISTIC_FILE = sys.argv[1]
FEATURE_FILE = sys.argv[2]
output_file = sys.argv[3]

#read feature scale
feature_scale = []
with open(FEATEURE_STATISTIC_FILE) as fin:
	feature_scale = json.load(fin)
feature_num = len(feature_scale)

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
			scale = feature_scale[idx][1] - feature_scale[idx][0]
			if scale > 0:
				feature[idx] = (value - feature_scale[idx][0])/ scale * 2 - 1
			else:
				feature[idx] = value
		for i in range(feature_num):
			fout.write(' '+ str(i+1) + ':' + str(feature[i]))
		fout.write('\n')

