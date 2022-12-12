import numpy as np

class Expert:
    "Expert base class"
    rng = None
    def __init__(self, conf) -> None:
        "Initialize expert accuracy and setup configuration"
        self.accuracy = conf.accuracy if conf.accuracy else None
        self.n_labels = conf.n_labels
        self.conf = conf
        Expert.rng = conf.rng


class ExpertReal(Expert):
    """Expert for real data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        # Directory of expert predictions
        self.root_dir = conf.ROOT_DIR
        self.confusion_matrix = self.create_confusion_matrix()
        self.w_matrix = self.get_w_from_confusion_matrix() 

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

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

class ExpertRealMoreExpressive(Expert):
    """Expert for real data experiments using a more expressive context including the difficulty of samples"""
    def __init__(self, conf, y_groups) -> None:
        super().__init__(conf)
        self.root_dir = conf.ROOT_DIR
        self.y_groups = y_groups
        # Array of conditional confusion matrices, one per difficulty group 
        self.confusion_matrix = self.create_confusion_matrix()
        # Array of w_matrices, one for each conditional confusion matrix
        self.w_matrix = self.get_w_from_confusion_matrix()

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def create_confusion_matrix(self):
        with open(f"{self.root_dir}/expert/cifar10h-probs.npy", "rb") as f:
            cm_per_sample = np.load(f)

        with open(f"{self.root_dir}/data/human_model_truth_cifar10h.csv", "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            y = csv[:,-1].astype(int) - 1
        
        groups = set(self.y_groups)
        cm = np.zeros(shape=(len(groups), self.n_labels,self.n_labels))
        
        for g in groups:
            for i in range(self.n_labels):
                idx = np.argwhere((y == i) & (self.y_groups == g)).flatten()
                cm[g,i] = cm_per_sample[idx].mean(axis=0)
        return cm

class ExpertSynthetic(Expert):
    """Expert for synthetic data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.confusion_matrix = self.create_confusion_matrix(conf.class_probabilities, conf.is_oblivious)
        self.w_matrix = self.get_w_from_confusion_matrix()
   
    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def create_confusion_matrix(self, class_probs, is_oblivious):
        if is_oblivious:
            return np.ones(shape=(self.n_labels, self.n_labels))*(1/self.n_labels)

        a = class_probs
        ind = list(range(self.n_labels))
        uniform_sol = self.accuracy
        # Assign first the uniform solution for each element of the diagonal of the confusion matrix (CM)
        x = np.ones(self.n_labels)*uniform_sol
        # Reassign random mass
        while  len(ind) >= 2:
            # Pick random pairs of the diagonal
            idx1  = Expert.rng.choice(ind)
            ind.remove(idx1)
            idx2  = Expert.rng.choice(ind)
            ind.remove(idx2)
            
            # Set normalization term
            tmp = idx1
            idx1 = idx1 if a[0][idx1] > a[0][idx2] else idx2
            idx2 = idx2 if tmp==idx1 else tmp
            ratio = a[0][idx2]/a[0][idx1]

            # Move random mass from one element to another, while keeping the CM valid
            epsilon = Expert.rng.uniform(0,np.minimum((1 - uniform_sol)*ratio, uniform_sol*ratio ))
            x[idx2] = uniform_sol - epsilon
            x[idx1] = uniform_sol + epsilon*ratio

        if len(ind):
            x[ind[0]] = uniform_sol
        
        assert (x < 1).all() and (x > 0).all() 
        
        self.better_than_random = True if (x >= 1/self.n_labels).any() else False
        
        cm = np.zeros(shape=(self.n_labels,self.n_labels))
        for i,ac in enumerate(x):
            # Compute the uniform solution to the off diagonal elements
            uniform_sol = (1 - ac)/(self.n_labels - 1) 
            indices = list(range(i))+list(range(i+1,self.n_labels))
            cm[i,i] = ac
            # Assign random mass to the off diagonal elements using 
            # random perturbations of the uniform solution
            while  len(indices) >= 2:
                # Pick random pairs of the off diagonal elements
                idx1 = Expert.rng.choice(indices)
                indices.remove(idx1)
                idx2 = Expert.rng.choice(indices)
                indices.remove(idx2)

                # Move random mass from the one element to the other
                epsilon = Expert.rng.normal(0, uniform_sol/6)
                cm[i,idx1] = uniform_sol - epsilon
                cm[i,idx2] = uniform_sol + epsilon

            if len(indices):
                cm[i,indices[0]] = uniform_sol
        
        # Diagnostics to confirm that the CM is valid  
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
        assert prob_faults==0 and exp_faults==0 and acc_errors==0

        return cm