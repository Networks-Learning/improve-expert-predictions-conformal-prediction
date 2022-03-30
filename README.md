# Provably Improving Expert Predictions with Conformal Prediction

## Install dependences

Experiments ran on python 3.7.3, with GPU. To install the required libraries, set the torch version according to the available device (CPU or GPU) in `requirements.txt` and run:

`pip install -r requirements.txt`

## Running experiments

For synthetic experiments run:

`python3 ./run_conf_synthetic.py --n_labels `<*n*\> `--cal_split` <*split*\> `--runs` <*runs*\>

where:
*  <*n*\> is the the number of labels, i.e. $n$. 
* <*split*\> is the calibration and estimation split, i.e. $\frac{m}{data\_to\_split}$.
* <*runs*\> is the number of times that each experiment will run with different random splits of the above specified size.

**Note:** the above runs experiments for  $\mathbb{P}[\hat Y = Y | \mathcal{Y}]\in\{0.3,0.5,0.7,0.9\}$ and classifiers' accuracies also $\in\{0.3,0.5,0.7,0.9\}$.

For real data experiments run:

`python3 ./run_conf_real.py --cal_split` <*split*\> `--runs` <*runs*\>

where <*split*\> and <*runs*\> are the same as above.

**Note:** the above runs experiments for all classifiers in the paper.

## Results

* All plots are produced in `plots.ipynb`. 
* For the Tables we used the functions `print_accuracy_synthetic()` and `print_accuracy_tables_real()` for synthetic and real data experiments results respectively, in `plot/plot.py`. 
* For results regarding the relative gain in success probability $\mathbb{P}[\hat{Y}= Y| \mathcal{C}_{\hat{\alpha}}(X)]$ with respect to $\mathbb{P}[\hat{Y}= Y| \mathcal{Y}]$ we used `get_mn()` for synthetic experiments and `get_m_real()` for real data experiments in `plot/plot.py`. 
