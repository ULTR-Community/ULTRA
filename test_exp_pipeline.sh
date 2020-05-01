# Clean
rm -r ./tmp_model/
rm -r ./tmp_output/
#rm -r ../dist/
#rm -r ../build/

# Run model
python main.py --max_train_iteration=100 --setting_file=./example/offline_setting/dla_exp_settings.json

# Test model
python main.py --test_only=True