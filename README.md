# Detecting clinician implicit biases in diagnoses using proximal causal inference [[paper]](https://psb.stanford.edu/psb-online/proceedings/psb25/liu_k.pdf)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)

## Overview

We provide a user-friendly tool to detect implicit biases in observational datasets. This database provides example application to synthetic data as Jupyter notebooks under `notebooks`. Application to real data should follow similarly, although feature curation and analysis is on a case-by-case basis. 

The main class of our method is `ProximalDE` (found in `proximalde.proximal.py`), which calculates the implicit bias direct effect and provides access to all the auxiliary tests used to validate the result. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Application to Real Data](#data)
- [Contributing](#contributing)
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
Given a real dataset, we can compute a semi-synthetic dataset with known implicit bias effect $c = \theta$, as detailed in our paper (although in this notebook, we use pure synthetic data as the input dataset). The `SemiSyntheticGenerator` object is described in detail in `proximalde.gen_synthetic_data.py` and can be used like such: 
```
generator = SemiSyntheticGenerator(random_state=0,split=True)
generator.fit(W, D, Z, X, Y, ZXYres=None) 
Wtilde, Dtilde, _, Ztilde, Xtilde, Ytilde = generator.sample(nsamples, a, b, c, g, replace=True)
```
#### `Tests.ipynb`
#### `SyntheticViolationExploration.ipynb`
#### `CustomRegressionModels.ipynb`

### Application to Real Data 

Provide instructions and examples on how to use the project.

```bash
python main.py
```

## Features

- Feature 1: Brief description
- Feature 2: Brief description
- Feature 3: Brief description

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact:

- Name: Kara Liu
- Email: karaliu [at] stanford . edu
