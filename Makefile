init:
		pip install -r requirements.txt

format:
		autopep8 --in-place --aggressive -r ./ultra/

test:
		pytest tests/ --ignore=tests/inte_tests/ --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc
		# flake8 ./matchzoo --exclude __init__.py
