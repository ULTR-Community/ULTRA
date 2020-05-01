import os
import sys

SVM_RANK_PATH = sys.argv[1]  # the directory of the SVMrank program
TRAIN_FILE = sys.argv[2]  # the training file path
VALID_FILE = sys.argv[3]  # the validation file path
TEST_FILE = sys.argv[4]  # the test file path
OUTPUT_PATH = sys.argv[5]  # the output directory

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# train model
train_command = SVM_RANK_PATH + 'svm_rank_learn -c 200 ' + \
    TRAIN_FILE + ' ' + OUTPUT_PATH + 'model.dat'
print(train_command)
os.system(train_command)

# test model
command = SVM_RANK_PATH + 'svm_rank_classify ' + TRAIN_FILE + \
    ' ' + OUTPUT_PATH + 'model.dat ' + OUTPUT_PATH + 'train.predict'
print(command)
os.system(command)
command = SVM_RANK_PATH + 'svm_rank_classify ' + VALID_FILE + \
    ' ' + OUTPUT_PATH + 'model.dat ' + OUTPUT_PATH + 'valid.predict'
print(command)
os.system(command)
command = SVM_RANK_PATH + 'svm_rank_classify ' + TEST_FILE + \
    ' ' + OUTPUT_PATH + 'model.dat ' + OUTPUT_PATH + 'test.predict'
print(command)
os.system(command)
