## Build Documentation:



#### Install Requirements

```python
pip install -r requirements.txt
```



#### Build Documentation

```python
# Enter docs folder.
cd docsource
# Use sphinx autodoc to generate rst.
# usage: sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN,...]
sphinx-apidoc -o source/ ../ultra/
# Generate html from rst
make clean
make github
```

