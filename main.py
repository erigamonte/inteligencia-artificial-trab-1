
import argparse
from src.problem.regression_problem import RegressionProblem
from src.problem.classification_problem import ClassificationProblem
from src.problem.tsp_problem import TSPProblem
from src.algorithm.genetic_algorithm import genetic_algorithm
import numpy as np
import pandas as pd


def generate_report(output_best_fitness, output_bests_fitness_generation, output_time):
    df = pd.DataFrame({
        'max': [np.max(output_best_fitness)],
        'min': [np.min(output_best_fitness)],
        'media': [np.mean(output_best_fitness)],
        'd. padrao': [np.std(output_best_fitness)],
        'tempo total (s)': [np.mean(output_time)]
    })
    print(df)
    
def build_problem(problem_name):
    if problem_name == "classification":
        return ClassificationProblem("data/german_statlog/german.data-numeric")
    elif problem_name == "regression":
        return RegressionProblem("data/regression/data-3.txt")
    elif problem_name == "tsp":
        return TSPProblem("data/tsp/tsp-30.txt")
    else:
        raise NotImplementedError()


def read_command_line_args():
    parser = argparse.ArgumentParser(
        description='Optimization with genetic algorithms.')

    parser.add_argument('-p', '--problem', default='tsp',
                        choices=["classification", "regression", "tsp"])
    parser.add_argument('-n', '--n_generations', type=int,
                        default=1000, help='number of generations.')
    parser.add_argument('-s', '--population_size', type=int,
                        default=20, help='population size.')
    parser.add_argument('-m', '--mutation_rate', type=float,
                        default=0.1, help='mutation rate.')

    args = parser.parse_args()
    return args


def main():
    args = read_command_line_args()

    problem = build_problem(args.problem)

    output_best_fitness, output_bests_fitness_generation, output_time = [], [], []

    for _ in range(5):
        bf, bfg, t = [], [], -1

        bf, bfg, t = genetic_algorithm(
            problem,
            population_size=args.population_size,
            n_generations=args.n_generations,
            mutation_rate=args.mutation_rate)

        output_best_fitness.append(bf)
        output_bests_fitness_generation.append(bfg)
        output_time.append(t)

    generate_report(output_best_fitness, output_bests_fitness_generation, output_time)

    print("OK!")

if __name__ == "__main__":
    main()
