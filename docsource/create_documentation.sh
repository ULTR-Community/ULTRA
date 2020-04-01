# Regenerate rst files

python ultra/input_layer/parameter_readme_generator.py
python ultra/learning_algorithm/parameter_readme_generator.py
python ultra/ranking_model/parameter_readme_generator.py

cd docs
sphinx-apidoc -o source/ ../ultra

# make html files
make clean
make html