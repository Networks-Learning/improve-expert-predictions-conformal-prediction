import numpy as np
import torch
import os
import argparse

class Config:
    def __init__(self) -> None:
        pass
    
parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", type=int, default=10, help='Size of label space in synthetic prediction tasks') 
parser.add_argument("--cal_split", type=float, help='Fraction of data to be used for the estimation and calibration set') 
parser.add_argument("--runs", type=int, default=10, help='Number of repetitions of each experiment')
parser.add_argument("--topk", type=int, default=5, help='Set "k" for prediction sets with top-k labels')
args,unknown = parser.parse_known_args()
conf = Config()

conf.ROOT_DIR = os.path.dirname(__file__)

if torch.cuda.is_available():
    conf.device = torch.cuda.current_device()
else:
    conf.device = 'cpu'

conf.seed = 12345678
conf.torch_rng = torch.Generator(device=conf.device).manual_seed(conf.seed)
conf.rng = np.random.default_rng(seed=conf.seed)

# Dataset size
conf.data_size = 10000 
# Parameter to control difficulty of the dataset in synthetic experiments
conf.class_sep = {10:{0.3:0.46, 0.5:1.09, 0.7:1.72, 0.9: 2.75}, 
                  50:{0.3:1.31, 0.5:2.16, 0.7:3.19, 0.9: 5.27},
                 100:{0.3:1.75, 0.5:2.8, 0.7:4.4, 0.9: 7.7}}

conf.accuracies = np.arange(3,10, 2)/10.
conf.is_oblivious = False # If set, human predicts labels at random

conf.n_labels = args.n_labels
conf.cal_split = args.cal_split

conf.test_split = 0.2 # Test split for synthetic data experiments
conf.n_runs_per_split = args.runs
conf.delta = 0.1
# Synthetic data label distribution
distr = conf.rng.dirichlet(np.ones(conf.n_labels),size=1)
sum_distr = distr.sum()
if sum_distr < 1.:
    distr += (1 - sum_distr)/conf.n_labels
conf.class_probabilities = distr
# Names of classifiers used in real data experiments
conf.model_names = ['densenet-bc-L190-k40','preresnet-110','resnet-110']