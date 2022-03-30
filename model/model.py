from sklearn.linear_model import LogisticRegression
from config import conf
import numpy as np
import torch

class Model:
    def __init__(self) -> None:
        pass

    def train(self, x,y):
        pass

    def predict(self, input):
        pass
    
    def predict_prob(self, input):
        pass
    
    def test(self, x, y):
        pass

class ModelReal(Model):
    """Model used in real data experiments"""
    def __init__(self, m_name) -> None:
        super().__init__()
        # 'm_name' specifies if we are using DenseNet, PreResNet-110 or ResNet-110.
        with open(f"{conf.ROOT_DIR}/data/{m_name}.csv", "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            self.model_logits = csv[:, 11:] 
            # models keep stored the softmax outputs for each sample
            # so they need only the index of the correspondent sample to return the softmax output

    def predict(self, input, return_tensor=False):
        self.model_logits_t = torch.tensor(self.model_logits[input], device=conf.device)
        y_hat = self.model_logits_t.multinomial(1, replacement=True, generator=conf.torch_rng)
        if not return_tensor:
            y_hat = y_hat.detach().cpu().numpy().flatten() 
        return y_hat
    
    def predict_prob(self, input):
        return self.model_logits[input]

    def test(self, x, y):
        y_hat = self.predict(input=x)
        return np.mean(y == y_hat)



class ModelSynthetic(Model):
    """Model used in synthetic data experiments"""
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(random_state=0,n_jobs=-1, max_iter=1000, multi_class='ovr')
        self.missing_classes = []
        
    def predict(self, input):
        return self.model.predict(input)

    def predict_prob(self, input):
        ret = self.model.predict_proba(input)
        # fix model output size with 0 probabilty for unknown classes
        for missing_class in self.missing_classes:

            ret = np.insert(ret,missing_class,0.,axis=1)
    

        return ret

    def train(self, x,y):
        self.model = self.model.fit(x,y)
        # Find which classes the model did not learn at all 
        # (Needed to fix tensors size later)
        sorted_classes = np.sort(self.model.classes_)
        all_classes = np.arange(conf.n_labels)
        if self.model.classes_.shape[0] < conf.n_labels:
            i =0
            for j in all_classes:
                if j == sorted_classes[i]:
                    i+=1
                else:
                    self.missing_classes.append(j)




    def test(self, x, y):
        return self.model.score(x,y)