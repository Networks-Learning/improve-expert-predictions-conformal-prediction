import numpy as np
from collections import defaultdict


class Expert:
    rng = None
    def __init__(self, conf) -> None:
        "Initialize expert and expert's confusion matrix"
        self.accuracy = conf.accuracy if conf.accuracy else None
        self.n_labels = conf.n_labels
        Expert.rng = conf.rng

class ExpertReal(Expert):
    """Expert for real data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.root_dir = conf.ROOT_DIR
        self.confusion_matrix = self.create_confusion_matrix()
        self.w_matrix = self.get_w_from_confusion_matrix()

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix) + 5

    def create_confusion_matrix(self):
        with open(f"{self.root_dir}/expert/cifar10h-probs.npy", "rb") as f:
            cm_per_sample = np.load(f)

        with open(f"{self.root_dir}/data/human_model_truth_cifar10h.csv", "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            y = csv[:,-1].astype(int) - 1
        
        
        cm = np.zeros(shape=(self.n_labels,self.n_labels))
        for i in range(self.n_labels):
            idx = np.argwhere(y == i).flatten()
            cm[i] = cm_per_sample[idx].mean(axis=0)
        return cm


class ExpertSynthetic(Expert):
    """Expert for synthetic data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        
        self.confusion_matrix = self.create_confusion_matrix(conf.class_probabilities, conf.is_oblivious)
        self.w_matrix = self.get_w_from_confusion_matrix()

        self.z_matrix = np.identity(n=self.n_labels)

    
    def pred_prob_given_y_C(self, y, label_set, alpha,update=True):
        """Probabilities of predicting each label from a prediction set given the true label y"""
        pred_prob = np.zeros(self.n_labels)
        denom = np.sum([np.exp(self.w_matrix[y][l]) for l in label_set])
        for label in label_set:
            pred_prob[label] = np.exp(self.w_matrix[y,label]) / denom

        return pred_prob


    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix) + 5

    def create_confusion_matrix(self, class_probs, is_oblivious):
        
        if is_oblivious:
            return np.ones(shape=(self.n_labels,self.n_labels))*(1/self.n_labels)
        a = class_probs

       
        ind = list(range(self.n_labels))
        uniform_sol = self.accuracy
        # assign first the uniform solution for each element of the diagonal of the confusion matrix (CM)
        x = np.ones(self.n_labels)*uniform_sol
        while  len(ind) >= 2:
            # pick random pairs of the diagonal
            idx1  = Expert.rng.choice(ind)
            ind.remove(idx1)
            idx2  = Expert.rng.choice(ind)
            ind.remove(idx2)
            
            # set normalization term
            tmp = idx1
            idx1 = idx1 if a[0][idx1] > a[0][idx2] else idx2
            idx2 = idx2 if tmp==idx1 else tmp
            ratio = a[0][idx2]/a[0][idx1]

            # move random mass from one element to another, while keeping the CM valid
            epsilon = Expert.rng.uniform(0,np.minimum((1 - uniform_sol)*ratio, uniform_sol*ratio ))
            x[idx2] = uniform_sol - epsilon
            x[idx1] = uniform_sol + epsilon*ratio


        if len(ind):
            x[ind[0]] = uniform_sol
        
        assert (x < 1).all() and (x > 0).all() 
        
        self.better_than_random = True if (x >= 1/self.n_labels).any() else False
        
        cm = np.zeros(shape=(self.n_labels,self.n_labels))
        for i,ac in enumerate(x):
            # assign the uniform solution to the off diagonal elements
            uniform_sol = (1 - ac)/(self.n_labels - 1) 
            indices = list(range(i))+list(range(i+1,self.n_labels))

            cm[i,i] = ac
            while  len(indices) >= 2:
                # pick random pairs of the off diagonal elements
                idx1 = Expert.rng.choice(indices)
                indices.remove(idx1)
                idx2 = Expert.rng.choice(indices)
                indices.remove(idx2)
                # move random mass from the one element to the other
                epsilon = Expert.rng.normal(0,  uniform_sol/6)
                cm[i,idx1] = uniform_sol - epsilon
                cm[i,idx2] = uniform_sol + epsilon
                

            if len(indices):
                cm[i,indices[0]] = uniform_sol
        # diagnostic to confirm that the CM is valid  
        prob_faults = 0
        exp_faults = 0
        for j in range(self.n_labels):
            prob_faults += np.abs(sum(cm[j,:]) - 1) >= .001
            exp_faults += sum(cm[:,j] * a[0]) >= 1

        acc_errors = 0
        s = 0
        for j in range(self.n_labels):
            s += cm[j, j] * a[0][j]
        acc_errors += np.abs(s - self.accuracy) >= .01
        print(prob_faults, exp_faults, acc_errors)
        return cm