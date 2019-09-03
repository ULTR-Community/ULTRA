# ULTRA
This is an Unbiased Learning To Rank Algorithms (ULTRA) toolbox, which is still UNDER DEVELOPMENT.

Please cite the following paper when you use this toolbox:

> Ai, Qingyao, Jiaxin Mao, Yiqun Liu, and W. Bruce Croft. "Unbiased learning to rank: Theory and practice." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 2305-2306. ACM, 2018.

## Install

**Create virtual environment (optional):**
```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

**Install ULTRA from the source:**
```
git clone https://github.com/QingyaoAi/ULTRA.git
cd ULTRA
python setup.py install #use setup-gpu.py for GPU support
```

**Run toy example:**
```
cd example/toy
bash offline_exp_pipeline.sh
```

## Get Started


## Algorithms

1. [DLA](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/dla.py): this is an implementation of the Dual Learning Algorithm in <a href="https://arxiv.org/pdf/1804.05938.pdf">Unbiased Learning to Rank with Unbiased Propensity Estimation</a>.

2. [IPW](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/ipw_rank.py): this model is an implementation of the Inverse Propensity Weighting algorithms in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf"> Learning to Rank with Selection Bias in Personal Search</a> and <a href="https://arxiv.org/pdf/1608.04468.pdf"> Unbiased Learning-to-Rank with Biased Feedback</a>

3. [REM](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/regression_EM.py): this model is an implementation of the regression-based EM algorithm in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46485.pdf">Position bias estimation for unbiased learning to rank in personal search</a>

4. [PD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/pairwise_debias.py): this model is an implementation of the pairwise debiasing algorithm in <a href="https://arxiv.org/pdf/1809.05818.pdf">Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm.</a>

5. [PDGD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/pdgd.py): this model is an implementation of the Pairwise Differentiable Gradient Descent algorithm in <a href="https://arxiv.org/abs/1809.08415">Differentiable unbiased online learning to rank</a>

6. [DBGD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/dbgd.py): this model is an implementation of the Dual Bandit Gradient Descent algorithm in <a href="https://arxiv.org/abs/1503.03244">Interactively optimizing information retrieval systems as a dueling bandits problem</a>

7. [NA](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/na.py): this model is an implementation of the naive algorithm that directly train models with clicks.

## Citation

## Project Organizers