cd ../../

# Download SVMrank
wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz

tar xvzf svm_rank_linux64.tar.gz

# you may need to sort features and remove duplicated features
python ./utils/libsvm/initial_ranking_with_svm_rank.py ./ ./example/toy/data/train.txt ./example/toy/data/valid.txt ./example/toy/data/test.txt ./tmp/

python ./utils/libsvm/prepare_exp_data_with_svmrank.py ./example/toy/data/ ./tmp/ ./tmp_data/ 136

# run model
python main.py