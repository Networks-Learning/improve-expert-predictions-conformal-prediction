# improve-expert-predictions-conformal-prediction
This is a repository containing the code used in the paper [Provably Improving Expert Predictions with Conformal Prediction](https://arxiv.org/abs/2201.12006).

## Install Dependences

Experiments ran on python 3.7.3, with GPU. To install the required libraries, set the torch version according to the available device (CPU or GPU) in `requirements.txt` and run:

```pip install -r requirements.txt```

## Code Structure

### Automated Decision Support System using Conformal Prediction

`conformal_prediction.py` contains the implementation of the conformal prediction based decision support system with the following key components:

* `find_all_alpha_values` finds all values of $\alpha$ to be examined, using the calibration set 
* `find_alpha_star` finds the near optimal $\alpha$ value, based on the estimation of the conditional expert's success probability using the estimation set (works for standard and modified conformal prediction). 

### Experimental Setup

* `config.py` contains the configuration details for both the synthetic and real data experiments.
* `utils.py` contains functions for generating/reading the synthetic and real datasets.
* `./model/model.py` contains the classes for the models used in the synthetic and real data experiments.
* `./expert/expert.py` contains the classes for the experts used in the synthetic and real data experiments.


### Scripts Executing Experiments
* `run_conf_synthetic.py` runs the synthetic data experiments. It takes as parameters the label space size, the calibration and estimation sets split, and the number of each experiment's repetitions using random splits. It runs experiments for all combinations of experts and models with success probabilities $\in\{0.3,0.5,0.7,0.9\}$.
* `run_conf_real.py` runs the real data experiments. It takes as parameters  the calibration and estimation sets split, and the number of each experiment's repetitions using random splits. It runs experiments for all the 3 combinations of the expert with the pre-trained models.

### Evaluation and Plots
The above scripts compute and store also the average human misprediction rate on test set, while using the decision support system. Moreover:
* `synthetic_avg_size.py` computes and stores the average prediction set size for each of the examined values of $\alpha$ in synthetic data experiments. It takes the same parameters as `run_conf_synthetic.py`.
* `real_avg_size.py` copmutes and stores the average prediction set size for each of the examined values of $\alpha$ in real data experiments. It takes the same parameters as `run_conf_real.py`.
* `./plot/plot.py`contains all plotters and the functions that compute all the numerical results reported in the paper.
* `plots.ipynb` produces the figures appeared in the paper.


## Running Experiments

For **synthetic** data experiments run:

`python3 ./run_conf_synthetic.py --n_labels `<*n*\> `--cal_split` <*split*\> `--runs` <*runs*\>

where:
*  <*n*\> is the total number of labels. 
* <*split*\> is the calibration and estimation split.
* <*runs*\> is the number of times that each experiment will run with different random splits of the above specified size.

Results will be stored under `./results_synthetic`.


For **real** data experiments run (<*split*\> and <*runs*\> same as above):

`python3 ./run_conf_real.py --cal_split` <*split*\> `--runs` <*runs*\>

 Results will be stored under `./results_real`.


## Citation
If you use parts of the code in this repository for your own research, please consider citing:

```
@article{straitouri22provably,
         title={Provably Improving Expert Predictions with Conformal Prediction},
         author={Straitouri, Eleni and Wang, Lequn and Okati, Nastaran and Rodriguez, Manuel Gomez},
         journal={arXiv preprint arXiv:2201.12006},
         year={2022}
         
}
```
