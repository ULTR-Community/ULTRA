cd ../../

# Download Yahoo! Letor dataset.
# wget https://webscope.sandbox.yahoo.com/download.php?r=39201&d=
# The link might be expired. If so, please go to https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=64 for more information.
# tar -zxvf Webscope_C14B.tgz
cd ./Webscope_C14B
tar -zxvf ltrc_yahoo.tgz
cd ../

# Prepare the dataset "set 1".
# Sort features, sort query id, remove duplicates, and remove queries without relevant documents in validation and test set.
python ./libsvm_tools/clean_libsvm_file.py ./Webscope_C14B/set1.train.txt ./Webscope_C14B/train.txt 0
python ./libsvm_tools/clean_libsvm_file.py ./Webscope_C14B/set1.valid.txt ./Webscope_C14B/valid.txt 1
python ./libsvm_tools/clean_libsvm_file.py ./Webscope_C14B/set1.test.txt ./Webscope_C14B/test.txt 1

# Sample 1% training data to build the initial ranker.
python ./libsvm_tools/sample_libsvm_data.py ./Webscope_C14B/train.txt ./Webscope_C14B/sampled_train.txt 0.01

# Download SVMrank.
wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz
tar xvzf svm_rank_linux64.tar.gz

# Conduct initial ranking with SVMrank.
python ./libsvm_tools/initial_ranking_with_svm_rank.py \
    ./ \
    ./Webscope_C14B/sampled_train.txt \
    ./Webscope_C14B/valid.txt \
    ./Webscope_C14B/test.txt \
    ./tmp/
./svm_rank_classify ./Webscope_C14B/train.txt ./tmp/model.dat ./tmp/train.predict

# Prepare model input.
python ./libsvm_tools/prepare_exp_data_with_svmrank.py ./Webscope_C14B/ ./tmp/ ./tmp_data/ 700

# run model
python main.py --setting_file=./example/offline_setting/dla_exp_settings.json