cd ../../

# Download SVMrank
wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz

tar xvzf svm_rank_linux64.tar.gz

# You may need to sort features and remove duplicated features
python ./utils/libsvm/initial_ranking_with_svm_rank.py ./ ./example/toy/data/train.txt ./example/toy/data/valid.txt ./example/toy/data/test.txt ./tmp/

python ./utils/libsvm/prepare_exp_data_with_svmrank.py ./example/toy/data/ ./tmp/ ./tmp_data/ 136

# Run model
python main.py --max_train_iteration=1000 --setting_file=./example/offline_setting/dla_exp_settings.json

# Test model
python main.py --test_only=True
