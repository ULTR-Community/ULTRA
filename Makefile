init:
		pip install -r requirements.txt

format:
		autopep8 --in-place --aggressive -r ./ultra/
		autopep8 --in-place --aggressive -r ./tests/

test:
		pytest tests/ --ignore=tests/inte_tests/ 
