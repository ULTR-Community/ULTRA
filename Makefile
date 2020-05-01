init:
		pip install -r requirements.txt

format:
		autopep8 --in-place --aggressive -r ./ultra/ --exclude="parameter_readme_generator.py"
		autopep8 --in-place --aggressive -r ./tests/

codecov:
		codecov -t token

TEST_ARGS = -v --full-trace -l --cov ultra/ --cov-report term-missing --cov-report xml --cov-config .coveragerc ultra/ tests/ -W ignore::DeprecationWarning --ignore=tests/inte_tests/ 
test:
		pytest $(TEST_ARGS)
