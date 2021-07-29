

import numpy as np
import random
import matplotlib.pyplot as plt

from src.problem.problem_interface import ProblemInterface


class TSPProblem(ProblemInterface):
    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()
        
        # For each line l, remove the "\n" at the end using
        # rstrip and then split the string the spaces. After
        # this instruction, lines is a list in which each element
        # is a sublist of strings [s1, s2, s3, ..., sn].
        lines = [l.rstrip().rsplit() for l in lines]
        
        # Convert the list of list into a numpy matrix of floats.
        self.cities = np.array(lines).astype(np.float)

    def distance(self, i_city_1, i_city_2):
        city1 = self.cities[i_city_1]
        city2 = self.cities[i_city_2]

        # distancia = diferenca euclidiana
        return (((city1[0] - city2[0]) ** 2) + (city1[1] - city2[1]) ** 2) ** 0.5

    def fitness(self, individual):
        dist = 0

        # soma a distancia entre cada ponto
        for i in range(1, len(individual)):
            dist += self.distance(individual[i-1], individual[i])
        
        # inclui o calculo da origem ate a segunda cidade e a distancia da ultima cidade ate a origem
        return self.distance(0, individual[0]) + dist + self.distance(individual[-1], 0)

    def new_individual(self):
        individual = list(range(1, len(self.cities)))
        np.random.shuffle(individual)
        return individual

    def mutation(self, individual):
        # tratamento por causa que a passagem eh por parametro no python
        individual = np.copy(individual)

        # obtem aleatoriamente a quantidade de vezes que sera feita a mutacao
        n_changes = np.random.randint(1, 3)

        for i in range(n_changes):
            # obtem posicoes aleatorias do vetor para realizar a troca de posicoes
            pos = random.sample(range(0, len(individual)), 2)
            # troca os elementos
            individual[pos[0]], individual[pos[1]] = individual[pos[1]], individual[pos[0]]

        return individual

    def cross_order_one(self, parent, child, begin):
        index_parent = begin
        index_child = begin
        
        while (0 in child):
            if(parent[index_parent] not in child):
                child[index_child] = parent[index_parent]
                index_child+=1
                index_parent+=1
            else:
                index_parent+=1
            
            if(index_child >= len(child)):
                index_child = 0
            if(index_parent >= len(parent)):
                index_parent = 0

        # print(child)
        # print('index_child: ' + str(index_child))
        # print('index_parent: ' + str(index_parent))
        # print('------------')

        return child

    def crossover(self, p1, p2):
        # inicia os vetores que serao armazenados os filhos
        c1 = np.zeros((len(p1), ), dtype=np.int)
        c2 = np.zeros((len(p2), ), dtype=np.int)

        # gera pivos aleatorios para fazer o corte
        pos = random.sample(range(1, len(p1)-2), 2)
        i1, i2 = pos[0], pos[1]

        # verifica se os pivos sao validos
        # o i2 tem que ser MAIOR que o i1
        if(i1 > i2):
            i1, i2 = i2, i1

        # faz a troca de valores que estao entre os pivos do pai1 com o pai2
        for i in range(i1, i2):
            c1[i] = p1[i]
            c2[i] = p2[i]
        
        # print('index1: ' + str(i1))
        # print('index2: ' + str(i2))
        # print('p1: ' + str(p1))
        # print('p2: ' + str(p2))
        # print('c1: ' + str(c1))
        # print('c2: ' + str(c2))

        return self.cross_order_one(p1, c2, i2), self.cross_order_one(p2, c1, i2)

    def plot(self, individual):
        x, y = [], []

        x.append(self.cities[0][0])
        y.append(self.cities[0][1])
        
        for i in range(0, len(individual)):
            x.append(self.cities[individual[i]][0])
            y.append(self.cities[individual[i]][1])
        
        x.append(self.cities[0][0])
        y.append(self.cities[0][1])

        # plt.ion()
        # plt.show()
        
        # # limpa grafico antetior
        # plt.cla()

        # # informa novos pontos a serem desenhados
        # plt.plot(x, y, "-bo", markerfacecolor="r")

        # #desenha o grafico
        # plt.draw()
        # plt.pause(0.000001)

        
