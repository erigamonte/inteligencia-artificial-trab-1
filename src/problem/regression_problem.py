
from src.problem.problem_interface import ProblemInterface
import numpy as np
import math 
import matplotlib.pyplot as plt

class RegressionProblem(ProblemInterface):
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
        self.points = np.array(lines).astype(np.float)
        
        # fixado n = 4
        self.n = 4
        # fixado coeficiente = 9
        self.number_coefficient = 9
        # fixado range de coeficiente = [-100,100]
        self.range_coefficient = [-100, 100]

    def function_fx(self, individual, x):
        total = individual[0]
        a_pos = 1
        b_pos = 2
        for i in range (1, self.n+1):
            total += individual[a_pos]*math.sin(i*x)+individual[b_pos]*math.cos(i*x)
            a_pos += 2
            b_pos += 2

        return total

    def fitness(self, individual):
        total = 0
        for i in range (0, len(self.points)):
            total += (self.points[i][1] - self.function_fx(individual, self.points[i][0])) ** 2

        return total * 1/len(self.points)

    def new_individual(self):
        # a1 = individual[0], b1 = individual[1], ..., an = individual[n-2], bn = individual[n-1]
        individual = []

        for _ in range(0, self.number_coefficient):
            individual.append(np.random.uniform(self.range_coefficient[0], self.range_coefficient[1]))
        return individual

    def mutation(self, individual):
        # tratamento por causa que a passagem eh por parametro no python
        individual = np.copy(individual)

        # obtem aleatoriamente a quantidade de vezes que sera feita a mutacao
        n_changes = np.random.randint(1, 3)

        for i in range(n_changes):
            # obtem uma posicao aleatoria do vetor para realizar a mutacao
            pos = np.random.randint(0, len(individual))

            # em 50% dos casos altera por um outro valor, nos outros 50% soma o valor de uma normal de desvio padrao 1
            if(np.random.uniform(0,1) > 0.5):
                individual[pos] = np.random.uniform(self.range_coefficient[0], self.range_coefficient[1])
            else:
                v_normal = np.random.normal(scale=1)
                # verifica se a soma ira ultrapassar o limite superior, se ultrapassar nao eh somado nenhum valor
                individual[pos] +=  v_normal if individual[pos] + v_normal > self.range_coefficient[1] else 0
            
        return individual

    def crossover(self, p1, p2):
        c1, c2 = [], []
        
        v_alfa = np.random.uniform(0, 1)

        for i in range(len(p1)):
            c1.append(p1[i]*v_alfa + (1-v_alfa)*p2[i])
            c2.append(p2[i]*v_alfa + (1-v_alfa)*p1[i])

        return c1, c2

    def plot(self, individual):

        x_points, y_points = [], []
        
        # for i in range(0, len(self.points)):
        #     x_points.append(self.points[i][0])
        #     y_points.append(self.points[i][1])

        # x = np.linspace(np.min(x_points), np.max(x_points), num=100)
        # y = []
        # for i in range(0, len(x)):
        #     y.append(self.function_fx(individual, x[i]))
        
        # plt.ion()
        # plt.show()
        
        # # limpa grafico anterior
        # plt.cla()

        # # informa novos pontos a serem desenhados
        # plt.plot(x_points, y_points, "or")

        # plt.plot(x, y, "-b")

        # #desenha o grafico
        # plt.draw()
        # plt.pause(0.000001)
