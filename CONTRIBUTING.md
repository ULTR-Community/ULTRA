Contributing to ULTRA
----------

> Note: ULTRA is developed under Python 3.6.

Welcome! ULTRA is a community project that aims to support research on unbiased learning to rank and online learning to rank. Your experience and what you can contribute are important to the project's success.

Discussion
----------

If you run into unexpected behavior in ULTRA, have trouble on applying it to your code, or find a bug or a feature you would like to get, please do not hesitate to contact us!

The main forum for discussion is the project's [GitHub issue tracker](https://github.com/ULTR-Community/ULTRA/issues).  This is the right place to start a discussion of any of the above or most any other topic concerning the project.

First Time Contributors
-----------------------

ULTRA appreciates your contribution! If you are interested in contributing to ULTRA, there are several ways to get started:

* Work on [new data simulation/preparation paradigm](https://github.com/ULTR-Community/ULTRA/tree/master/ultra/input_layer).
* Work on [new unbiased/online learning algorithm](https://github.com/ULTR-Community/ULTRA/tree/master/ultra/learning_algorithm).
* Work on [new learning-to-rank model](https://github.com/ULTR-Community/ULTRA/tree/master/ultra/ranking_model).
* Work on [new ranking metric/utils](https://github.com/ULTR-Community/ULTRA/tree/master/ultra/utils).
* Work on [documentation](https://github.com/ULTR-Community/ULTRA/tree/master/docsource).
* Try to answer questions on [the issue tracker](https://github.com/ULTR-Community/ULTRA/issues).

Submitting Changes
------------------

We use the standard GitHub pull-request flow, which may be familiar to you if you've contributed to other projects on GitHub -- see blow. 

Anyone interested in ULTRA may review your code.  One of the core developers will check and merge your pull request when they think it's ready.
If you go a few days without a reply, please feel free to ping the thread by adding a new comment.

A list of ULTRA core developers can be found in [Readme](https://github.com/ULTR-Community/ULTRA/blob/master/README.md).

Contributing Flow
------------------

1. Fork the latest version of [ULTRA](https://github.com/ULTR-Community/ULTRA/) into your repo.
2. Create an issue under [ULTR-Community/ULTRA/](https://github.com/ULTR-Community/ULTRA/issues), write description about the bug/enhancement.
3. Clone your forked ULTRA into your machine, add your changes. 
4. Create json files in tests/test_settings/ (e.g.,tests/test_settings/test.json) to test models that have been changed. Please create separate json files to test each changed model.
5. Run `make test` to ensure all tests passed on your computer.
6. Run `make format` to use autopep8 to format your code.
7. Push to your forked repo, then send the pull request to the official repo. In pull request, you need to create a link to the issue you created using `#[issue_id]`, and describe what has been changed.
8. We'll assign reviewers to review your code.


Your PR will be merged if:
- Funcitonally benefit for the project.
- With proper docstrings, see codebase as examples.
- All reviewers approved your changes.


**Thanks and let's improve ULTRA together!**