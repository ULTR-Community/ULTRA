Data_path="./istella-full-letor_trial"   ## Data path where to unzip the data
Data_folder="full"            ## subfolder after unzip
Feature_number=220              ## how many features for LETOR data
Prepro_fun="log"                ## additional function to do preprocessing, available, "log", "None", we default normalize data to -1 and 1. If choosing log, it will first using log function to the data and then normalize it to -1 and 1. 
prefix=""                       ## name before data, for example setl.train.txt, prefix=set1.
# Data_zip_file=istella-s-letor.tar.gz  ## zipped data file path.
Data_zip_file=is_full_trial_exp.tar.gz
cd ../../
# Download Istella-s dataset.
# wget http://library.istella.it/dataset/istella-s-letor.tar.gz    
# create path that will be used.
mkdir $Data_path
mkdir $Data_path/cleaned_data  # path to store data after cleaning
mkdir $Data_path/normalized    # path to store data after nomalization
mkdir $Data_path/tmp_toy       # path to store toy version of training data which is 1% of total dataset
mkdir $Data_path/tmp_toy/data
mkdir $Data_path/tmp_toy/tmp
mkdir $Data_path/tmp_toy/tmp_data_toy

tar -zxvf  $Data_zip_file --directory $Data_path
# Prepare the dataset.
# Sort features, sort query id, remove duplicates, and remove queries without relevant documents in validation and test set.
valid_name=$Data_path/$Data_folder/${prefix}vali.txt
if [ ! -f "$valid_name" ]
then
    echo "no vali, try to find valid"
    valid_name=$Data_path/$Data_folder/${prefix}valid.txt
    if [ ! -f "$valid_name" ]
    then
        echo "no valid, we will split trian with default rate"
        mv $Data_path/$Data_folder/${prefix}train.txt $Data_path/$Data_folder/${prefix}train_orig.txt
        python ./ultra/utils/libsvm/split_libsvm_data.py $Data_path/$Data_folder/${prefix}train_orig.txt ${valid_name} $Data_path/$Data_folder/${prefix}train.txt 0.1
    fi
fi
echo "begin cleaning"
python ./ultra/utils/libsvm/clean_libsvm_file.py $Data_path/$Data_folder/${prefix}train.txt $Data_path/cleaned_data/train.txt 0
python ./ultra/utils/libsvm/clean_libsvm_file.py ${valid_name}  $Data_path/cleaned_data/valid.txt 1
python ./ultra/utils/libsvm/clean_libsvm_file.py $Data_path/$Data_folder/${prefix}test.txt $Data_path/cleaned_data/test.txt 1
# Normalize the Istella-s data, since it is so unbalanced.
# Extract the feature statistics for later normalization.
echo "extract statistics for normalization"
python ./ultra/utils/libsvm/extrac_feature_statistics.py $Data_path/cleaned_data/
# Normalize the data.
echo "begin normalization"
python ./ultra/utils/libsvm/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/test.txt $Data_path/normalized/test.txt $Prepro_fun
python ./ultra/utils/libsvm/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/train.txt $Data_path/normalized/train.txt $Prepro_fun
python ./ultra/utils/libsvm/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/valid.txt $Data_path/normalized/valid.txt $Prepro_fun
# Sample 1% training data to build the initial ranker.
echo "sample 0.01 for intiial ranker"
python ./ultra/utils/libsvm/sample_libsvm_data.py $Data_path/normalized/train.txt $Data_path/normalized/sampled_train.txt 0.01

# Download SVMrank.
wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz
tar xvzf svm_rank_linux64.tar.gz

# Conduct initial ranking with SVMrank.
python ./ultra/utils/libsvm/initial_ranking_with_svm_rank.py \
    ./ \
    $Data_path/normalized/sampled_train.txt \
    $Data_path/normalized/valid.txt \
    $Data_path/normalized/test.txt \
    $Data_path/tmp/
./svm_rank_classify $Data_path/normalized/train.txt $Data_path/tmp/model.dat $Data_path/tmp/train.predict

# Prepare model input.
python ./ultra/utils/libsvm/prepare_exp_data_with_svmrank.py $Data_path/normalized/ $Data_path/tmp/ $Data_path/tmp_data/ $Feature_number


cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/train.txt
cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/valid.txt
cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/test.txt
./svm_rank_classify $Data_path/tmp_toy/data/train.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/train.predict
./svm_rank_classify $Data_path/tmp_toy/data/valid.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/valid.predict
./svm_rank_classify $Data_path/tmp_toy/data/test.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/test.predict
python ./ultra/utils/libsvm/prepare_exp_data_with_svmrank.py $Data_path/tmp_toy/data/ $Data_path/tmp_toy/tmp/ $Data_path/tmp_toy/tmp_data_toy/ $Feature_number



export SETTING_ARGS="--data_dir=$Data_path/tmp_data/ --model_dir=$Data_path/tmp_model/ --output_dir=$Data_path/tmp_output/ --setting_file=./example/offline_setting/dla_exp_settings.json"
echo $SETTING_ARGS
# Run model
python main.py --max_train_iteration=1000 $SETTING_ARGS

Test model
python main.py --test_only=True $SETTING_ARGS