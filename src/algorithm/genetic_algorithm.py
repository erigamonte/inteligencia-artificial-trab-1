import numpy as np
import copy

def genetic_algorithm(problem, population_size, n_generations, mutation_rate=0.1):

    population = create_new_population(problem, population_size)
    best_fitness, best_individual, worst_fitness_position = population_fitness(problem, population)

    for n in range(n_generations):
        new_population = []
        for i in range(population_size//2):
            p1 = tournament_selection(problem, population)
            p2 = tournament_selection(problem, population)
            
            # print('p1: ' + str(p1) + ' fitness: ' + str(problem.fitness(p1)))
            # print('p2: ' + str(p2) + ' fitness: ' + str(problem.fitness(p2)))
            
            c1, c2 = problem.crossover(p1, p2)

            # print('c1: ' + str(c1) + ' fitness: ' + str(problem.fitness(c1)))
            # print('c2: ' + str(c2) + ' fitness: ' + str(problem.fitness(c2)))

            if (np.random.uniform(0,1) > mutation_rate):
                c1 = problem.mutation(c1)
                # print('c1 - pos mutation: ' + str(c1) + ' fitness: ' + str(problem.fitness(c1)))

            if (np.random.uniform(0,1) > mutation_rate):
                c2 = problem.mutation(c2)
                # print('c2 - pos mutation: ' + str(c2) + ' fitness: ' + str(problem.fitness(c2)))

            new_population.append(c1)
            new_population.append(c2)

        best_fitness_new_pop, best_individual_new_pop, worst_fitness_position = population_fitness(problem, new_population)

        if(best_fitness_new_pop < best_fitness):
            best_individual = best_individual_new_pop
            best_fitness = best_fitness_new_pop

        # substitui o individuo de pior fitness pelo melhor 
        new_population[worst_fitness_position] = best_individual

        population = new_population

        problem.plot(best_individual)

        # print("iteration: " + str(n) + ' fitness: ' + str(best_fitness) + ' ' + str(best_individual))
        print("iteration: " + str(n) + ' fitness: ' + str(best_fitness))
        
    return best_individual

def create_new_population(problem, population_size):
    new_population = []
    for i in range(population_size):
        new_population.append(problem.new_individual())
    return new_population

def population_fitness(problem, population):
    best_fitness = problem.fitness(population[0])
    best_individual = population[0]
    worst_fitness = best_fitness
    worst_fitness_position = 0

    for i in range(1, len(population)):
        curr_fitness = problem.fitness(population[i])
        if(best_fitness > curr_fitness):
            best_individual = copy.deepcopy(population[i])
            best_fitness = curr_fitness
        if(worst_fitness < curr_fitness):
            worst_fitness = curr_fitness
            worst_fitness_position = i
    
    return best_fitness, best_individual, worst_fitness_position

def tournament_selection(problem, population):
    # obtem posicoes aleatorias dos pais
    i1 = np.random.randint(0, len(population))
    i2 = np.random.randint(0, len(population))
    if(problem.fitness(population[i1]) > problem.fitness(population[i2])):
        return population[i1]
    return population[i2]