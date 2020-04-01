****************************
Learning Algorithm Reference
****************************

DLA
###

Description
***********

The Dual Learning Algorithm for unbiased learning to rank.

This class implements the Dual Learning Algorithm (DLA) based on the input layer 
feed. See the following paper for more information on the algorithm.

* Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

Hyper Parameters
****************

IPWrank
#######

Description
***********

The Inverse Propensity Weighting algorithm for unbiased learning to rank.

This class implements the training and testing of the Inverse Propensity Weighting algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

* Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
* Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

Hyper Parameters
****************

RegressionEM
############

Description
***********

The regression-based EM algorithm for unbiased learning to rank.

This class implements the regression-based EM algorithm based on the input layer 
feed. See the following paper for more information.

* Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

In particular, we use the online EM algorithm for the parameter estimations:

* Cappé, Olivier, and Eric Moulines. "Online expectation–maximization algorithm for latent data models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.

Hyper Parameters
****************

PDGD
####

Description
***********

The Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

This class implements the Pairwise Differentiable Gradient Descent (PDGD) algorithm based on the input layer 
feed. See the following paper for more information on the algorithm.

* Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.

Hyper Parameters
****************

DBGD
####

Description
***********

The Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

This class implements the Dueling Bandit Gradient Descent (DBGD) algorithm based on the input layer 
feed. See the following paper for more information on the algorithm.

* Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

Hyper Parameters
****************

PairDebias
##########

Description
***********

The Pairwise Debiasing algorithm for unbiased learning to rank.

This class implements the Pairwise Debiasing algorithm based on the input layer 
feed. See the following paper for more information on the algorithm.

* Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

Hyper Parameters
****************

NavieAlgorithm
##############

Description
***********

The navie algorithm that directly trains ranking models with input labels.

    

Hyper Parameters
****************

DBGDInterleave
##############

Description
***********

The Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

This class implements the Dueling Bandit Gradient Descent (DBGD) algorithm based on the input layer 
feed. See the following paper for more information on the algorithm.

* Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

Hyper Parameters
****************

