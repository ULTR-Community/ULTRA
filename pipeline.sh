wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz

tar xvzf svm_rank_linux64.tar.gz

# you may need to sort features and remove duplicated features
python ./utils/libsvm/initial_ranking_with_svm_rank.py ./ ./example/train.txt ./example/valid.txt ./example/test.txt ./tmp/

python ./utils/libsvm/prepare_exp_data.py ./example/ ./tmp/ ./tmp_data/ 700

# run model
python main.py