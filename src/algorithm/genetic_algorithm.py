import numpy as np
import pandas as pd
import random
import copy
import time
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

def genetic_algorithm(problem, population_size, n_generations, mutation_rate):

    t_begin = time.time()

    bests_fitness_generation = []
    population = create_new_population(problem, population_size)
    best_individual = population_fitness(population)
    best_individual_prvl_generation = best_individual
    
    for n in range(n_generations):
        new_population = []
        for i in range(population_size//2):
            p1 = proportionate_selection(population)
            p2 = proportionate_selection(population)
            
            # print('p1: ' + str(p1) + ' fitness: ' + str(problem.fitness(p1)))
            # print('p2: ' + str(p2) + ' fitness: ' + str(problem.fitness(p2)))
            
            c1, c2 = problem.crossover(p1.solution, p2.solution)

            # print('c1: ' + str(c1) + ' fitness: ' + str(problem.fitness(c1)))
            # print('c2: ' + str(c2) + ' fitness: ' + str(problem.fitness(c2)))

            if (mutation_rate >= np.random.uniform(0,1)):
                c1 = problem.mutation(c1)
                # print('c1 - pos mutation: ' + str(c1) + ' fitness: ' + str(problem.fitness(c1)))

            if (mutation_rate >= np.random.uniform(0,1)):
                c2 = problem.mutation(c2)
                # print('c2 - pos mutation: ' + str(c2) + ' fitness: ' + str(problem.fitness(c2)))

            new_population.append(Individual(c1, problem.fitness(c1)))
            new_population.append(Individual(c2, problem.fitness(c2)))

        best_individual_new_pop = population_fitness(new_population)

        # guarda os melhores individuos de cada geracao
        bests_fitness_generation.append(best_individual_new_pop.fitness)

        # verifica se o melhor individuo é o melhor de todas as geracoes
        if(best_individual_new_pop.fitness < best_individual.fitness):
            best_individual = best_individual_new_pop

        # aplica o elitismo
        population = elitism(population, new_population, population_size)

        # SESSAO COMENTADA PARA O ELITISMO DE TROCA DO MELHOR DA GERACAO ANTERIOR POR UM ALEATORIO DA NOVA
        # ALTEREI PARA OS MELHORES DE AMBAS GERACOES APOS SUGESTAO EM AULA
        
        ########### substitui um individuo aleatorio pelo melhor da geracao anterior
        ########### change_pos = np.random.randint(0, len(new_population))
        ########### new_population[change_pos] = best_individual_prvl_generation

        ########### # armazena o melhor individuo da geracao atual para ser usado na proxima geracao
        ########### best_individual_prvl_generation = best_individual_new_pop

        ########### population = new_population

        # print("iteration: " + str(n) + ' bestfitness_gen: ' + str(best_individual_new_pop.fitness) + ' bestfitness: ' + str(best_individual.fitness))

    t_end = time.time()

    return best_individual, bests_fitness_generation, t_end-t_begin

def plot_best_solution(problem, output_bests_solution):
    output_bests_solution.sort(key=lambda x: x.fitness)
    for s in output_bests_solution:
        print('Fitness: ' + str(s.fitness))
        problem.plot(s.solution)

def plot_history(bests_fitness_generation, n_generations):
    for i in range(len(bests_fitness_generation)):
        x = list(range(1, n_generations+1))
        y = bests_fitness_generation[i]
        plt.plot(x, y, "-b")

    plt.ion()
    plt.show()
    #desenha o grafico
    plt.draw()
    plt.pause(0.000001)

def generate_report(output_best_solutions, output_time):
    output_best_fitness = [bs.fitness for bs in output_best_solutions]
    df = pd.DataFrame({
        'max': [np.max(output_best_fitness)],
        'min': [np.min(output_best_fitness)],
        'media': [np.mean(output_best_fitness)],
        'd. padrao': [np.std(output_best_fitness)],
        'tempo total (s)': [np.mean(output_time)]
    })
    print(df)

def elitism(population, new_population, population_size):
    # une a populacao anterior e a nova
    elitism_population = population + new_population
    # ordena por fitness
    elitism_population.sort(key=lambda x: x.fitness)
    #gera uma populacao com as melhores de ambas
    return elitism_population[:population_size]

def create_new_population(problem, population_size):
    new_population = []
    for i in range(population_size):
        ind_solution = problem.new_individual()
        new_population.append(Individual(ind_solution, problem.fitness(ind_solution)))
    return new_population

def population_fitness(population):
    best_individual = population[0]

    for i in range(1, len(population)):
        if(best_individual.fitness > population[i].fitness):
            # guarda uma copia profunda para nao ter perigo de em algum outro lugar ser modificado
            best_individual = copy.deepcopy(population[i])
    
    return best_individual

def tournament_selection(population):
    # obtem posicoes aleatorias dos pais
    solutions = random.sample(range(0, len(population)), 2)
    if(population[solutions[0]].fitness < population[solutions[1]].fitness):
        return population[solutions[0]]
    return population[solutions[1]]

def proportionate_selection(population):
    fitness = [p.fitness for p in population]
    min = np.min(fitness)

    # tratamento para caso a fitness seja negativa
    # busca o menor valor negativo e soma todos os valores com o oposto deste valor
    if(min < 0):
        for i in range(0,len(fitness)):
            fitness[i] = fitness[i] + min * (-1)

    max = np.max(fitness)
    
    total = 0
    # como a avaliacao do fitness deve ser o melhor valor sendo o menor
    # eh necessario verificar a diferenca do max com o valor do fitness
    total = sum(max - f for f in fitness)
   
    pick = random.uniform(0, total)
    current = 0
    for i in range(0, len(fitness)):
        current += max - fitness[i]
        if current >= pick:
            break

    return population[i]

