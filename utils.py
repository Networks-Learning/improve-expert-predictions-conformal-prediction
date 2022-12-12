from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from config import conf
import numpy as np

def make_dataset(run_no, machine_accuracy):
    """Synthetic dataset generation"""
    # Class separation parameter to control the difficulty of 
    # the dataset, and so the accuracy of the classifier
    clas_sep = conf.class_sep[conf.n_labels][machine_accuracy]

    x, y = make_classification(n_samples=conf.data_size, n_features=20, n_classes=conf.n_labels, n_informative=15,\
        n_redundant=5, class_sep=clas_sep, flip_y=0, weights=conf.class_probabilities[0], random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=conf.test_split, random_state=42+run_no)

    X_train, X_cal_est, y_train, y_cal_est = train_test_split(
         X_train, y_train, test_size=2*conf.cal_split, random_state=42+run_no)
    
    # Estimation and calibration sets have the same size
    X_cal, X_est, y_cal, y_est = train_test_split(
         X_cal_est, y_cal_est, test_size=0.5, random_state=42+run_no)

    return X_train, X_test,X_cal,X_est, y_train, y_test,y_cal, y_est 

def make_dataset_real(run_no):
    """Real dataset"""
    file_ground_truth = 'densenet-bc-L190-k40'
    with open(f"{conf.ROOT_DIR}/data/{file_ground_truth}.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')
        # Ground truth labels
        y = csv[:,0].astype(int)
        # Models need only the index of the sample as input
        x = np.arange(10000) 

    X_test, X_cal_est, y_test, y_cal_est = train_test_split(
        x, y, test_size=2*conf.cal_split, random_state=42+run_no)
    
    # Estimation and calibration sets have the same size
    X_cal, X_est, y_cal, y_est = train_test_split(
         X_cal_est, y_cal_est, test_size=0.5, random_state=42+run_no)
    
    return X_test, X_cal, X_est, y_test, y_cal, y_est

def make_dataset_real_with_difficulties(run_no):
    """Real dataset assuming different levels of difficulty across samples"""
    file_ground_truth = 'densenet-bc-L190-k40'
    with open(f"{conf.ROOT_DIR}/data/{file_ground_truth}.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')
        # Ground truth labels
        y = csv[:,0].astype(int)
        # Models need only the index of the sample as input
        x = np.arange(10000) 

    # Each sample is assinged a difficulty level
    y_groups = find_difficulties()
    # Y labels extended with the difficulty groups
    y_with_dif = np.array(list(zip(y, y_groups)))
    
    X_test, X_cal_est, y_test, y_cal_est = train_test_split(
        x, y_with_dif, test_size=2*conf.cal_split, random_state=42+run_no)
    
    # Estimation and calibration sets have the same size
    X_cal, X_est, y_cal, y_est = train_test_split(
         X_cal_est, y_cal_est, test_size=0.5, random_state=42+run_no)
    
    return X_test, X_cal, X_est, y_test, y_cal, y_est, y_groups

def find_difficulties():
    """Define difficulty levels of all samples"""
    with open(f"{conf.ROOT_DIR}/expert/cifar10h-probs.npy", "rb") as f:
        cm_per_sample = np.load(f)

    with open(f"{conf.ROOT_DIR}/data/human_model_truth_cifar10h.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')
        y = csv[:,-1].astype(int) - 1
    
    acc_per_sample = np.zeros(len(cm_per_sample))
    y_groups = np.zeros(len(cm_per_sample), dtype='int')
    # Average accuracy of experts for each sample
    for i,l in enumerate(y):
        acc_per_sample[i] = cm_per_sample[i][l]
    # Thresholds defining the difficulty levels
    threshold_hard = np.quantile(acc_per_sample, 0.25)
    threshold_easy = np.quantile(acc_per_sample, 0.5)

    for i,acc in enumerate(acc_per_sample):
        if acc > threshold_easy:
            # Easy samples
            y_groups[i] = 0
        elif acc < threshold_hard:
            # Hard samples
            y_groups[i] = 2
        else:
            # Medium difficulty samples
            y_groups[i] = 1
    
    return y_groups
 