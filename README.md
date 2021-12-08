<div align='center'>
<img src="https://github.com/ULTR-Community/ULTRA/blob/master/docs/logo.png?raw=true" width = "200"  alt="logo" align="center" />
</div>

# Unbiased Learning to Rank Algorithms (ULTRA)

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Documentation Status](https://readthedocs.org/projects/ultra/badge/?version=latest)](https://ultr-community.github.io/ULTRA/)
[![Build Status](https://travis-ci.org/ULTR-Community/ULTRA.svg?branch=master)](https://travis-ci.org/ULTR-Community/ULTRA/)
[![codecov](https://codecov.io/gh/ULTR-Community/ULTRA/branch/master/graph/badge.svg)](https://codecov.io/gh/ULTR-Community/ULTRA)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
<a href="https://twitter.com/intent/follow?screen_name=CommunityUltr">
        <img src="https://img.shields.io/twitter/follow/CommunityUltr?style=social&logo=twitter"
            alt="follow on Twitter"></a>
<!---[![Build Status](https://travis-ci.org/NTMC-Community/MatchZoo.svg?branch=master)](https://travis-ci.org/NTMC-Community/MatchZoo/)---> 
<!---[![codecov](https://codecov.io/gh/NTMC-Community/MatchZoo/branch/master/graph/badge.svg)](https://codecov.io/gh/NTMC-Community/MatchZoo)---> 
<!---[![Requirements Status](https://requires.io/github/NTMC-Community/MatchZoo/requirements.svg?branch=master)](https://requires.io/github/NTMC-Community/MatchZoo/requirements/?branch=master)---> 


🔥**News: A PyTorch version of this package can be found in [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch).**

This is an Unbiased Learning To Rank Algorithms (ULTRA) toolbox, which provides a codebase for experiments and research on learning to rank with human annotated or noisy labels. With the unified data processing pipeline, ULTRA supports multiple unbiased learning-to-rank algorithms, online learning-to-rank algorithms, neural learning-to-rank models, as well as different methods to use and simulate noisy labels (e.g., clicks) to train and test different algorithms/ranking models. A user-friendly documentation can be found [here](https://ultr-community.github.io/ULTRA/).

## Get Started

**Create virtual environment (optional):**
```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

**Install ULTRA from the source:**
```
git clone https://github.com/ULTR-Community/ULTRA.git
cd ULTRA
make init # Replace 'tensorflow' with 'tensorflow-gpu' in requirements.txt for GPU support
```

**Run toy example:**
```
bash example/toy/offline_exp_pipeline.sh
```

## Structure

<div align='center'>
<img src="https://github.com/ULTR-Community/ULTRA/blob/master/docs/structure.png?raw=true" width = "800"  alt="structure" align="center" />
</div>

### Input Layers

1. [ClickSimulationFeed](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/input_layer/click_simulation_feed.py): this is the input layer that generate synthetic clicks on fixed ranked lists to feed the learning algorithm.

2. [DeterministicOnlineSimulationFeed](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/input_layer/deterministic_online_simulation_feed.py): this is the input layer that first create ranked lists by sorting documents according to the current ranking model, and then generate synthetic clicks on the lists to feed the learning algorithm. It can do result interleaving if required by the learning algorithm.

3. [StochasticOnlineSimulationFeed](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/input_layer/stochastic_online_simulation_feed.py): this is the input layer that first create ranked lists by sampling documents based on their scores in the current ranking model and the Plackett-Luce distribution, and then generate synthetic clicks on the lists to feed the learning algorithm. It can do result interleaving if required by the learning algorithm.

4. [DirectLabelFeed](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/input_layer/direct_label_feed.py): this is the input layer that directly feed the labels of each documents (e.g., the true relevance labels or raw click logs) to the learning algorithm.

### Learning Algorithms

1. [NA](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/navie_algorithm.py): this model is an implementation of the naive algorithm that directly train models with input labels (e.g., clicks).

2. [DLA](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/dla.py): this is an implementation of the Dual Learning Algorithm in <a href="https://arxiv.org/pdf/1804.05938.pdf">*Unbiased Learning to Rank with Unbiased Propensity Estimation*</a>.

3. [IPW](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/ipw_rank.py): this model is an implementation of the Inverse Propensity Weighting algorithms in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf">*Learning to Rank with Selection Bias in Personal Search*</a> and <a href="https://arxiv.org/pdf/1608.04468.pdf"> *Unbiased Learning-to-Rank with Biased Feedback*</a>

4. [REM](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/regression_EM.py): this model is an implementation of the regression-based EM algorithm in <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46485.pdf">*Position bias estimation for unbiased learning to rank in personal search*</a>

5. [PD](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/pairwise_debias.py): this model is an implementation of the pairwise debiasing algorithm in <a href="https://arxiv.org/pdf/1809.05818.pdf">*Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm*.</a>

6. [DBGD](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/dbgd.py): this model is an implementation of the Dual Bandit Gradient Descent algorithm in <a href="https://www.cs.cornell.edu/people/tj/publications/yue_joachims_09a.pdf">*Interactively optimizing information retrieval systems as a dueling bandits problem*</a>

7. [MGD](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/mgd.py): this model is an implementation of the Multileave Gradient Descent in <a href="https://dl.acm.org/doi/pdf/10.1145/2835776.2835804">*Multileave Gradient Descent for Fast Online Learning to Rank*</a>

8. [NSGD](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/nsgd.py): this model is an implementation of the Null Space Gradient Descent algorithm in <a href="https://dl.acm.org/doi/pdf/10.1145/3209978.3210045">*Efficient Exploration of Gradient Space for Online Learning to Rank*</a>

9. [PDGD](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/learning_algorithm/pdgd.py): this model is an implementation of the Pairwise Differentiable Gradient Descent algorithm in <a href="https://arxiv.org/abs/1809.08415">*Differentiable unbiased online learning to rank*</a>

### Ranking Models

1. [Linear](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/ranking_model/Linear.py): this is a linear ranking algorithm that compute ranking scores with a linear function.

2. [DNN](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/ranking_model/DNN.py): this is neural ranking algorithm that compute ranking scores with a multi-layer perceptron network (with non-linear activation functions).

3. [DLCM](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/ranking_model/DLCM.py): this is an implementation of the Deep Listwise Context Model in <a href="https://arxiv.org/pdf/1804.05936.pdf">*Learning a Deep Listwise Context Model for Ranking Refinement*</a>.

4. [GSF](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/ranking_model/GSF.py): this is an implementation of the Groupwise Scoring Function in <a href="https://arxiv.org/pdf/1811.04415.pdf">*Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks*</a>.

5. [SetRank](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/ranking_model/SetRank.py): this is an implementation of the SetRank model in <a href="https://arxiv.org/abs/1912.05891">*SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval*</a>.

## Supported Evaluation Metrics

1. [MRR](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the <a href="https://en.wikipedia.org/wiki/Mean_reciprocal_rank">Mean Reciprocal Rank</a> (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

2. [ERR](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the Expected Reciprocal Rank from <a href="http://olivier.chapelle.cc/pub/err.pdf">*Expected reciprocal rank for graded relevance*</a>.

3. [ARP](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the Average Relevance Position (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

4. [NDCG](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Normalized Discounted Cumulative Gain</a> (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

5. [DCG](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Discounted Cumulative Gain</a> (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

6. [Precision](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the Precision (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

7. [MAP](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the <a href="https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision">Mean Average Precision</a> (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

8. [Ordered_Pair_Accuracy](https://github.com/ULTR-Community/ULTRA/blob/master/ultra/utils/metrics.py): the percentage of correctedly ordered pair (inherited from [TF-Ranking](https://github.com/tensorflow/ranking)).

## Click Simulation Example

**Create click models for click simulations**
```
python ultra/utils/click_models.py pbm 0.1 1 4 1.0 example/ClickModel
```
\* The output is a json file containing the click mode that could be used for click simulation. More details could be found in the code.

**(Optional) Estimate examination propensity with result randomization**
```
python ultra/utils/propensity_estimator.py example/ClickModel/pbm_0.1_1.0_4_1.0.json <DATA_DIR> example/PropensityEstimator/
```
\* The output is a json file containing the estimated examination propensity (used for IPW). DATA_DIR is the directory for the prepared data created by ./libsvm_tools/prepare_exp_data_with_svmrank.py. More details could be found in the code.

## Citation

If you use ULTRA in your research, please use the following BibTex entry.

```
@article{10.1145/3439861,
author = {Ai, Qingyao and Yang, Tao and Wang, Huazheng and Mao, Jiaxin},
title = {Unbiased Learning to Rank: Online or Offline?},
year = {2021},
issue_date = {February 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {39},
number = {2},
issn = {1046-8188},
url = {https://doi.org/10.1145/3439861},
doi = {10.1145/3439861},
journal = {ACM Trans. Inf. Syst.},
month = feb,
articleno = {21},
numpages = {29},
keywords = {unbiased learning, online learning, Learning to rank}
}

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


## Development Team

 ​ ​ ​ ​
<table border="0">
  <tbody>
    <tr align="center">
      <td>
        ​ <a href="https://github.com/QingyaoAi"><img width="50" height="50" src="https://github.com/QingyaoAi.png?s=50" alt="QingyaoAi"></a><br>
        ​ <a href="http://ir.aiqingyao.org/">Qingyao Ai</a> ​
        <p>Core Dev<br>
        ASST PROF, Univ. of Utah</p>​
      </td>
      <td>
         <a href="https://github.com/Taosheng-ty"><img width="50" height="50" src="https://github.com/Taosheng-ty.png?s=50" alt="Taosheng-ty"></a><br>
         <a href="https://github.com/Taosheng-ty">Tao Yang</a> ​
        <p>Core Dev<br> Ph.D., Univ. of Utah</p>​
      </td>
      <td>
        ​ <a href="https://github.com/huazhengwang"><img width="50" height="50" src="https://github.com/huazhengwang.png?s=50" alt="huazhengwang"></a><br>
         <a href="https://github.com/huazhengwang">Huazheng Wang</a>
         <p>Core Dev<br> Ph.D., Univ. of Virginia</p>​
      </td>
      <td>
        ​ <a href="https://github.com/defaultstr"><img width="50" height="50" src="https://github.com/defaultstr.png?s=50" alt="defaultstr"></a><br>
        ​ <a href="https://github.com/defaultstr">Jiaxin Mao</a>
        <p>Core Dev<br>
        Postdoc, Tsinghua Univ.</p>​
      </td>
    </tr>
  </tbody>
</table>


## Contribution

Please read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request. 


## Project Organizers

- Qingyao Ai
  * School of Computing, University of Utah
  * [Homepage](http://ir.aiqingyao.org/)


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2020-present, Qingyao Ai (QingyaoAi)
