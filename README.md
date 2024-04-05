# improve-expert-predictions-conformal-prediction
This is a repository containing the code used in the paper [Improving Expert Predictions with Prediction Sets](https://arxiv.org/abs/2201.12006), ICML 2023.

## Install Dependencies

Experiments ran on python 3.9.2, with GPU. To install the required libraries, set the torch version according to the available device (CPU or GPU) in `requirements.txt` and run:

```pip install -r requirements.txt```

## Code Structure

### Automated Decision Support System using Conformal Prediction

`conformal_prediction.py` contains functions to implement and evaluate our conformal prediction based decision support system, with the following key components:

* `find_all_alpha_values` finds all values of $\alpha$ to be examined, using the calibration set. 
* `find_alpha_star` finds the near optimal $\alpha$ value, based on the estimation of the expert's success probability (works for standard and modified conformal prediction). 

### Experimental Setup

* `config.py` contains the configuration details for both the synthetic and real data experiments.
* `utils.py` contains functions for generating/reading the synthetic and real datasets.
* `./model/model.py` contains the classes for the models used in the synthetic and real data experiments.
* `./expert/expert.py` contains the classes for the experts used in the synthetic and real data experiments.

### Scripts Executing Experiments

#### Single experiment scripts

Under `scripts/single/{synthetic|real}` are scripts for synthetic and real data experiments given the calibration data split, and in case of synthetic data, additionally given the label space size. 

Scripts for both synthetic and real data:
* `standard_cp.py` runs experiments using standard conformal prediction, given the calibration and estimation set splits and the number of each experiment's repetitions using random splits. In case of synthetic data, it takes additionally as argument the label space size, and runs experiments for all combinations of experts and models with success probabilities $\in\{0.3,0.5,0.7,0.9\}$. In case of real data, it runs experiments for all the pre-trained models.
* `avg_size.py` computes and stores the average prediction set size for each value of $\alpha$. It takes the same arguments and runs similarly to `standard_cp.py`.
* `coverage.py` computes the near optimal alpha value and then measures the empirical coverage during test. It takes the same arguments as `standard_cp.py`. 
* `topk.py` measures the empirical expert misprediction probability during test using a top $k$ predictor. It takes the same arguments as `standard_cp.py` and additionally takes the $k$ value. It runs similarly as `standard_cp.py`.
* `robustness_analysis.py` measures the empirical expert misprediction probability during test under IIA violations as described in Appendix E. It takes the same arguments and runs similarly to `standard_cp.py`.

Scripts only for synthetic data:
* `modified_cp.py` runs experiments using modified conformal prediction. It takes the same arguments and runs similarly to `standard_cp.py`.
* `set_size_distribution.py` computes the subset size distribution for the near optimal $\alpha$ value. It takes the same arguments and runs similarly to `scripts/single/synthetic/standard_cp.py`.

Scripts only for real data:
* `more_expressive.py` runs the experiments experiments using a discrete choice model with a more expressive context including the difficulty level of samples. It takes the same arguments and runs similarly to `scripts/single/real/standard_cp.py`.

#### Batch experiment scripts

Under `scripts/batch/{synthetic|real}` are scripts for running multiple synthetic and real data experiments for several calibration data splits, and in case of synthetic data, for several label space sizes. `scripts/batch/synthetic/run_all_splits_nlabels.py` runs `scripts/single/synthetic/standard_cp.py`, `scripts/batch/real/run_all_splits.py` runs `scripts/single/real/standard_cp.py`, and each `run_all_`<*script*\>`.py` runs the relevant <*script*\>.

### Numerical Results and Plots
Functions to produce plots and numerical results are in the below scripts. More specifically:

* `./plot/plot.py` contains all plotters to produce the figures appeared in the paper.
* `./plot/printers.py` contains all the functions that compute all the numerical results reported in the paper.
* `plots.ipynb` produces the figures appeared in the paper in presence of the results.


## Running Scripts

For **synthetic** data experiments run:

`python3 -m scripts.single.synthetic.standard_cp --n_labels `<*n*\> `--cal_split` <*split*\> `--runs` <*runs*\>

where:
*  <*n*\> is the total number of labels. 
* <*split*\> is the calibration and estimation split.
* <*runs*\> is the number of times that each experiment will run with different random splits of the above specified size.

Results will be stored under `./results_synthetic`. To run all experiments for <*n*\> $\in\{10,50,100\}$, <*split*\> $\in \{0.02, 0.05, 0.1, 0.15,\}$ and <*runs*\> $=10$, run:

`python3 ./scripts/batch/synthetic/run_all_splits_nlabels.py`


For **real** data experiments run (<*split*\> and <*runs*\> same as above):

`python3  -m scripts.single.real.standard_cp  --cal_split` <*split*\> `--runs` <*runs*\>

 Results will be stored under `./results_real`. To run all experiments for <*split*\> $\in \{0.02, 0.05, 0.1, 0.15,\}$ and <*runs*\> $=10$, run:

`python3 ./scripts/batch/real/run_all_splits.py`


## Citation
If you use parts of the code in this repository for your own research, please consider citing:

```
@inproceedings{straitouri23improving,
         title={Improving Expert Predictions with Prediction Sets},
         author={Straitouri, Eleni and Wang, Lequn and Okati, Nastaran and Gomez-Rodriguez, Manuel},
         booktitle={Proceedings of the 40th International Conference on Machine Learning},
         year={2023}
         
}
```
