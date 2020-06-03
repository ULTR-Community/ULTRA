init:
		pip install -r requirements.txt

format:
		autopep8 --in-place --aggressive -r ./ultra/ --exclude="parameter_readme_generator.py"
		autopep8 --in-place --aggressive -r ./tests/

doc:
		pip install -r ./docsource/requirements.txt
		bash ./docsource/create_documentation.sh

codecov:
		bash <(curl -s https://codecov.io/bash) -t $(cc_token)

TEST_ARGS = -v --full-trace -l --cov ultra/ --cov-report term-missing --cov-report xml --cov-config .coveragerc ultra/ tests/ -W ignore::DeprecationWarning --ignore=tests/inte_tests/ 
test:
		pytest $(TEST_ARGS)
