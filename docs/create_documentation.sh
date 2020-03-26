# Regenerate rst files

python ultra/learning_algorithm/parameter_readme_generator.py

cd docs
sphinx-apidoc -o source/ ../ultra

# make html files
make clean
make html