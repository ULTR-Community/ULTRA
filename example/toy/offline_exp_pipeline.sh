# Toy example
export SETTING_ARGS="--data_dir=./example/toy/data/ --model_dir=./tmp_model/ --output_dir=./tmp_output/ --setting_file=./example/offline_setting/dla_exp_settings.json"

# Run model
python main.py --max_train_iteration=10 $SETTING_ARGS

# Test model
python main.py --test_only=True $SETTING_ARGS
