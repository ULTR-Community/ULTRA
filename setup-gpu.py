import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf').read()


setup(
    name="ULTRA",
    version="0.1.0",
    author="Qingyao Ai",
    author_email="aiqingyao@gmail.com",
    description=("TODO"),
    license="BSD",
    keywords="TODO",
    url="TODO",
    packages=find_packages(),
    long_description='TODO'\
    'TODO',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache License",
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=[
        #'keras >= 2.0.5',
        'tensorflow-gpu >= 1.1.0,<2',
        'numpy >= 1.12.1',
        'six >= 1.10.0',
        'scipy >= 1.0.0',
        'autopep8 >= 1.0.0',
    ]
    #extras_require={
    #    'visualize': ['matplotlib >= 2.2.0'],
    #    'tests': [
    #        'coverage >= 4.3.4',
    #        'codecov >= 2.0.15',
    #        'pytest >= 3.0.3',
    #        'pytest-cov >= 2.4.0',
    #        'mock >= 2.0.0',
    #        'flake8 >= 3.2.1',
    #        'flake8_docstrings >= 1.0.2'],
    #}
)
