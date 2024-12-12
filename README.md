# Detecting clinician implicit biases in diagnoses using proximal causal inference [[paper]](https://psb.stanford.edu/psb-online/proceedings/psb25/liu_k.pdf)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)

## Overview

We provide a user-friendly tool to detect implicit biases in observational datasets. This database provides example application to synthetic data as Jupyter notebooks under `notebooks`. Application to real data should follow similarly, although feature curation and analysis is on a case-by-case basis. 

The main class of our method is `ProximalDE` (found in `proximalde.proximal.py`), which calculates the implicit bias direct effect and provides access to all the auxiliary tests used to validate the result. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    - [Examples via Jupyter Notebooks](#examples-via-jupyter-notebooks)
    - [Application to Real Data](#application-to-real-data)
- [License](#license)
- [Contact](#contact)

## Installation
The repository can be installed by running the following commands:
```
$ git clone https://github.com/syrgkanislab/hidden_mediators
$ cd hidden_mediators
$ pip install -r requirements.txt
$ python setup.py develop
```

## Usage 
The bulk of our method for estimating and evaluating implicit bias effects is in the `ProximalDE` class, which can be called as such:
```
estimator = ProximalDE(model_regression, model_classification, binary_D, binary_Z, binary_X, binary_Y, ivreg_type, alpha_multipliers, alpha_exponent, cv, semi, n_jobs, verbose, random_state)
```
where all the arguments are optional (see `proximalde.proximal.py` for further description on arguments and defaults). Briefly, `ProximalDE` initiates the specifics of the models used for residualizing $W$ or calculating the dual / primal solutions, as well as specifics of the data (e.g., which variables are continuous versus categorical). 

The estimator is then fit over a dataset to calculate the implicit bias effect $\theta$ by calling:
```
est.fit(W, D, Z, X, Y)
```
where $D$, $Y$ are N-dimensional vectors, $W,Z,X$ are matrices of size N by $\{p_X, p_Z, p_X\}$, and all features are as described in the paper (i.e., following the assumptions of the required causal graph). 

Results can then computed and accessed via: 
```
estimator.summary() #displays the tables
```
which will display three tables: (1) the table containing the point estimate, standard error, and confidence interval; (2) the table containing the R2 of each of the four models ($Y, D, Z, X$) that residualized $W$; and (3) the table of the test results for 4 / 5 proposed tests: the primal violation, the dual violation, and the strength identification tests. These tables can also be accessed individually: 
```
sm = estimator.summary()
point_table = sm.tables[0]
r2_table = sm.tables[1]
test_table = sm.tables[2]
```
To run the 5th test, the proxy covariance rank test, we can run 
```
svalues, svalues_crit = estimator.covariance_rank_test(calculate_critical=True)
```
Additional analysis tests are detailed in the notebooks and include: 
1. Running the weak IV confidence interval (`estimator.robust_conf_int(alpha=0.05)`)
2. Analyzing influence scores. We calculate the influence score of the estimate (as described in the paper) as well as other metrics like Cook's distance (`inf_diag = est.run_influence_diagnostics()`). After running `run_influence_diagnostics`, the size and effect of high-influence sets can be analyzed. 
3. Bootstrapped estimation by resampling and re-estimating at various stages (`estimator.bootstrap_inference(stage=stage, n_subsamples=n_subsamples, fraction=0.5)`)

Finally, we provide two additional algorithms we developed to aid in implicit bias estimation: semi-synthetic data generation and the proxy selection algorithm. We describe these further below. 

### Examples via Jupyter Notebooks
We provide several notebooks with example computation and analyses: 
#### `SyntheticExperiments.ipynb`
This notebooks presents the basics of how to run use the `ProximalDE` class and how to run the tests we provide to validate the results. This includes a very simple influence analysis, as well as subsample bootstrap experiments. 

Purely synthetic data can be generated for experimentation by specifying parameters into 
```
W, X, Z, D, Y = gen_data(a, b, c, d, e, f, g, pm, pz, px, pw, n, sm=sm, seed=seed)
```
Knowing the implicit bias effect $c = \theta$ also allows us to run many iterations of an experiment and to collect various metrics, like MSE and coverage. We show an example of this in this notebook. 

#### `SemiSyntheticExperiments.ipynb`
Given a real dataset, we can compute a semi-synthetic dataset with known implicit bias effect $c = \theta$, as detailed in our paper (although in this notebook, we naively use pure synthetic data as the input dataset). The `SemiSyntheticGenerator` object is described in detail in `proximalde.gen_synthetic_data.py` and can be used like such: 
```
generator = SemiSyntheticGenerator(random_state=0, test_size=.5, split=True)
generator.fit(W, D, Z, X, Y) 
What, Dhat, _, Zhat, Xhat, Yhat = generator.sample(nsamples, a, b, c, g)
```
#### `Tests.ipynb`
This notebook walks through each of the 5 tests we propose (dual violation, primal violation, strength identification tests, covariance rank test) and generates examples in which we expect and see that each test fails when violations in synthetic data are violated. For example, if we generate $X$ to be weak proxy without any relationship $M \rightarrow X$ or $X \rightarrow Y$, then we expect the primal violation test to fail. 
#### `SyntheticViolationExploration.ipynb`
In our work, we found using the entire set of available proxies $X$ and $Z$ leads to both the dual and primal violation tests failing. In this notebook, we work backwards essentially by simulating synthetic data that fails both the dual and primal tests.

Importantly, this notebook also introduces how to call and use the proposed proxy selection algorithm. The method entails the class `ProxySelection` (detailed in depth in `proximalde.proxy_selection_alg.py`), which can be called as such: 
```
psalg = ProxySelection(Xres,Zres,Dres,Yres,primal_type='full', violation_type='full',est_thresh=.05)
```
where all the arguments are described in depth in the `proximalde.proxy_selection_alg.py`. Calling the algorithm assumes that the data fails the dual and primal. 

This class takes the data after residualization of $W$ to find a list of candidate sets of proxy features in $X$ and $Z$ that pass the dual and primal tests. Candidate sets can be found by calling 
```
candidates = prm.find_candidate_sets(ntrials,niters=2)
```
which varies on thoroughness and time to compute candidates based on the number of `ntrials` selected (this is parameter $K$ in the paper; we typically used `ntrials`~=100). The result list `candidates` contains potential feature subsets of the proxies $X$ and $Z$ that should now pass the dual and primal tests. To confirm, a new estimator `ProximalDE` should be fit over the new proxies. For robustness, it is wise to run `ProxySelection` over one split of the dataset, and to evaluate `ProximalDE` on the other split of proxy subsets. 

#### `CustomRegressionModels.ipynb`
If user wants to try other models besides the available model options for residualizing $W$, we provide a notebook walking you through this. The main thing this model must inherit is the `BaseEstimator, RegressorMixin` classes (from sklearn). We provide an example `XGBRegressorWrapper` custom model that can then be passed into the `ProximalDE` class to use for residualizing $W$ (i.e., `ProximalDE(model_regression=XGBRegressionWrapper(), semi=False)`). 
### Application to Real Data 
Applying this method to real data requires first and foremost careful selection into which variables you are designating as $W, X, Z$, as well as knowledge about how $Y$ and $D$ were collected such that all causal assumptions hold. Variable selection is on a case-by-case basis per dataset. 

After variables have been selected, missingness should be analyzed and handled appropriately. Finally, the data can be grouped where $D$, $Y$ are N-dimensional vectors, $W,Z,X$ are matrices of size N by $\{p_W, p_Z, p_X\}$. A few suggestions we found when running on real data:
1. If $W,Z,X$ are high-dimensional (i.e. >50), repeatedly residualzing $W$ might be expensive, and it might be wise to save the residuals once and load automatically for usage. 
2. We recommend running `ProximalDE` over all the data before assuming the proxy selection algorithm should be used. If any of the tests fail, there likely needs to be a modification of variables (i.e., see the paper and `Tests.ipynb` for better intuition on how a test failure could inform how variables should be updated). Only if both the dual and primal fail should the proxy selection algorithm be run (and again, it should be done on a separate split of the data than the split used to evaluate the proxy subset with `ProximalDE`).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact:

- Name: Kara Liu
- Email: karaliu [at] stanford . edu
