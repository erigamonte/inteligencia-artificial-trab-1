import numpy as np
import random
import copy
import time

from pandas._libs.hashtable import value_count

class Individual:
    def __init__(self, value, fitness):
        self.value = value
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
            
            c1, c2 = problem.crossover(p1.value, p2.value)

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

        if(best_individual_new_pop.fitness < best_individual.fitness):
            best_individual = best_individual_new_pop

        # substitui um individuo aleatorio pelo melhor da geracao anterior
        change_pos = np.random.randint(0, len(new_population))
        new_population[change_pos] = best_individual_prvl_generation

        # armazzena o melhor individuo da geracao atual para ser usado na proxima geracao
        best_individual_prvl_generation = best_individual_new_pop
        
        # altera a populacao pela nova populacao
        population = new_population

        problem.plot(best_individual.value)

        print("iteration: " + str(n) + ' bestfitness_gen: ' + str(best_individual_new_pop.fitness) + ' bestfitness: ' + str(best_individual.fitness))

    t_end = time.time()

    return best_individual.fitness, bests_fitness_generation, t_end-t_begin

def create_new_population(problem, population_size):
    new_population = []
    for i in range(population_size):
        ind_value = problem.new_individual()
        new_population.append(Individual(ind_value, problem.fitness(ind_value)))
    return new_population

def population_fitness(population):
    best_individual = population[0]

    for i in range(1, len(population)):
        if(best_individual.fitness > population[i].fitness):
            best_individual = copy.deepcopy(population[i])
    
    return best_individual

def tournament_selection(population):
    # obtem posicoes aleatorias dos pais
    values = random.sample(range(0, len(population)), 2)
    if(population[values[0]].fitness < population[values[1]].fitness):
        return population[values[0]]
    return population[values[1]]

def proportionate_selection(population):
    min = np.min([p.fitness for p in population])
    #TODO somar o menor valor negativo caso existir

    max = np.max([p.fitness for p in population])
    
    total = 0
    total = sum(max-p.fitness for p in population)
   
    pick = random.uniform(0, total)
    current = 0
    for p in population:
        current += max - p.fitness
        if current >= pick:
            return p
