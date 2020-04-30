# Regenerate rst files

python ultra/input_layer/parameter_readme_generator.py
python ultra/learning_algorithm/parameter_readme_generator.py
python ultra/ranking_model/parameter_readme_generator.py

# Enter doc source folder.
cd docsource
# Use sphinx autodoc to generate rst.
# usage: sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN,...]
sphinx-apidoc -o source/ ../ultra

# make html files
make clean
make github

# add twitter card
python add_twitter_card.py ../docs/index.html ./twitter_card.json