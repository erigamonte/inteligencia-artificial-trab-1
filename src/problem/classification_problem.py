
import numpy as np
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.problem.problem_interface import ProblemInterface


class ClassificationProblem(ProblemInterface):
    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()

        # For each line l, remove the "\n" at the end using
        # rstrip and then split the string the spaces. After
        # this instruction, lines is a list in which each element
        # is a sublist of strings [s1, s2, s3, ..., sn].
        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.int32)

        # Split features (x) and labels (y). The notation x[:, i]
        # returns all values of column i. To learn more about numpy indexing
        # see https://numpy.org/doc/stable/reference/arrays.indexing.html .
        x = lines[:, :-1]
        y = lines[:, -1]

        # Split the data in two sets without intersection.
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(x, y, test_size=0.30,
                             stratify=y, random_state=871623)

        # number of features
        self.n_features = self.X_train.shape[1]

        # search space for the values of k and metric
        self.Ks = [1, 3, 5, 7, 9, 11, 13, 15]
        self.metrics = ["euclidean", "hamming", "canberra", "braycurtis"]

    def new_individual(self):
        individual = [np.random.randint(0, 2) for _ in range(0, len(self.X_train[0]))]
        ks_pos = np.random.randint(0, len(self.Ks))
        metrics_pos = np.random.randint(0, len(self.metrics))
        
        individual.append(self.Ks[ks_pos])
        individual.append(self.metrics[metrics_pos])

        return individual

    def fitness(self, individual):

        binary_pattern = individual[:-2]
        K = individual[-2]
        metric = individual[-1]

        # return the indices of the features that are not zero.
        indices = np.nonzero(binary_pattern)[0]

        # check if there is at least one feature available
        if len(indices) == 0:
            return 1e6

        # select a subset of columns given their indices
        x_tr = self.X_train[:, indices]
        x_val = self.X_val[:, indices]

        # build the classifier
        knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
        # train
        knn = knn.fit(x_tr, self.y_train)
        # predict the classes for the validation set
        y_pred = knn.predict(x_val)
        # measure the accuracy
        acc = np.mean(y_pred == self.y_val)

        # since the optimization algorithms minimize,
        # the fitness is defiend as the inverse of the accuracy
        fitness = -acc

        return fitness

    def mutation(self, individual):
        # tratamento por causa que a passagem eh por parametro no python
        individual = copy.deepcopy(individual)
        
        # em 50% dos casos altera os valores binarios, nos outros 50% faz a mutacao dos hyperparametros
        if(np.random.uniform(0,1) > 0.5):
            # obtem uma posicao aleatoria do vetor para realizar a mutacao
            pos = np.random.randint(0, len(individual)-2)
            individual[pos] = 0 if individual[pos] == 1 else 1
        else:
            pos = np.random.randint(1, 2)
            if(pos == 1):
                metrics_pos = np.random.randint(0, len(self.metrics))
                individual[-pos] = self.metrics[metrics_pos]
            else:
                ks_pos = np.random.randint(0, len(self.Ks))
                individual[-pos] = self.Ks[ks_pos]
            
        return individual

    def crossover(self, p1, p2):
        c1, c2 = [], [] 

        # 1 a len(p1)-2 para desconsiderar as posicoes do hyperparam e ter pelo menos um ponto de troca
        single_point_pos = np.random.randint(1, len(p1)-2)

        for i in range(len(p1)-2):
            if(i < single_point_pos):
                c1.append(p1[i])
                c2.append(p2[i])
            else:
                c1.append(p2[i])
                c2.append(p1[i])

        c1.append(p1[-2])
        c1.append(p2[-1])
        c2.append(p2[-2])
        c2.append(p1[-1])

        return c1, c2

    def plot(self, individual):
        pass
