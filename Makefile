init:
		pip install -r requirements.txt

format:
		autopep8 --in-place --aggressive -r ./ultra/
		autopep8 --in-place --aggressive -r ./tests/


TEST_ARGS = -v --full-trace -l --cov ultra/ --cov-report term-missing --cov-report html --cov-config .coveragerc ultra/ tests/ -W ignore::DeprecationWarning --ignore=tests/inte_tests/ 
test:
		pytest $(TEST_ARGS)
