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


## Ranking Models

1. [Linear](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/ranking_model/Linear.py): this is a linear ranking algorithm that compute ranking scores with a linear function.

2. [DNN](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/ranking_model/DNN.py): this is neural ranking algorithm that compute ranking scores with a multi-layer perceptron network (with non-linear activation functions).

3. [DLCM](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/DLCM.py): this is an implementation of the Deep Listwise Context Model in <a href="https://arxiv.org/pdf/1804.05936.pdf">*Learning a Deep Listwise Context Model for Ranking Refinement*</a>. (TODO)

4. [GFS](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/GFS.py): this is an implementation of the Groupwise Scoring Function in <a href="https://arxiv.org/pdf/1811.04415.pdf">*Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks*</a>. (TODO)

## Learning Algorithms

1. [DLA](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/dla.py): this is an implementation of the Dual Learning Algorithm in <a href="https://arxiv.org/pdf/1804.05938.pdf">*Unbiased Learning to Rank with Unbiased Propensity Estimation*</a>.

2. [IPW](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/ipw_rank.py): this model is an implementation of the Inverse Propensity Weighting algorithms in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf">*Learning to Rank with Selection Bias in Personal Search*</a> and <a href="https://arxiv.org/pdf/1608.04468.pdf"> *Unbiased Learning-to-Rank with Biased Feedback*</a>

3. [REM](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/regression_EM.py): this model is an implementation of the regression-based EM algorithm in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46485.pdf">*Position bias estimation for unbiased learning to rank in personal search*</a>

4. [PD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/pairwise_debias.py): this model is an implementation of the pairwise debiasing algorithm in <a href="https://arxiv.org/pdf/1809.05818.pdf">*Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm*.</a>

5. [PDGD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/pdgd.py): this model is an implementation of the Pairwise Differentiable Gradient Descent algorithm in <a href="https://arxiv.org/abs/1809.08415">*Differentiable unbiased online learning to rank*</a>

6. [DBGD](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/dbgd.py): this model is an implementation of the Dual Bandit Gradient Descent algorithm in <a href="https://arxiv.org/abs/1503.03244">*Interactively optimizing information retrieval systems as a dueling bandits problem*</a>

7. [NA](https://github.com/QingyaoAi/ULTRA/blob/master/learning_algorithm/na.py): this model is an implementation of the naive algorithm that directly train models with input labels (e.g., clicks).

## Supported Evaluation Metrics

## Click Simulation Guideline

**Create click models for click simulations**
```
python input_layer/click_models.py pbm 0.1 1 4 1.0 example/ClickModel
```
\* The output is a json file containing the click mode that could be used for click simulation. More details could be found in the code.

**(Optional) Estimate examination propensity with result randomization**
```
cd utils
python propensity_estimator.py ../example/ClickModel/pbm_0.1_1.0_4_1.0.json <DATA_DIR> ../example/PropensityEstimator/
```
\* The output is a json file containing the estimated examination propensity (used for IPW). More details could be found in the code.

## Citation

If you use ULTRA in your research, please use the following BibTex entry.

```
@inproceedings{Ai:2018:ULR:3269206.3274274,
 author = {Ai, Qingyao and Mao, Jiaxin and Liu, Yiqun and Croft, W. Bruce},
 title = {Unbiased Learning to Rank: Theory and Practice},
 booktitle = {Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '18},
 year = {2018},
 isbn = {978-1-4503-6014-2},
 location = {Torino, Italy},
 pages = {2305--2306},
 numpages = {2},
 url = {http://doi.acm.org/10.1145/3269206.3274274},
 doi = {10.1145/3269206.3274274},
 acmid = {3274274},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {click model, counterfactual learning, unbiased learning to rank, user bias},
} 
```

## Project Organizers
