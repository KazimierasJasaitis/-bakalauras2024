import pso_vectorized as hs
import pso_vectorized_new as ps
import imageCut
import csv
import numpy as np
import sys

# def run_hso(parametersGeneral, parametersHso):
#     hso = hs.HarmonySearch(paper_size=parametersGeneral["paper_size"],
#                         image_sizes=parametersGeneral["image_sizes"],
#                         dimensions=parametersGeneral["dimensions"],
#                         iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"],
#                         desired_fitness=parametersGeneral["desired_fitness"], 
#                         HM_size=parametersHso['HM_size'],
#                         memory_consideration_rate=parametersHso['memory_consideration_rate'],
#                         pitch_adjustment_rate=parametersHso['pitch_adjustment_rate'],
#                         pitch_bandwidth=parametersHso['pitch_bandwidth'])

#     best_position = hso.run()

#     with open("testResultsHso.csv", 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([parametersGeneral['id'], hso.best_fitness, hso.iterations])
    
#     with open("testSolutionsHso.csv", 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([parametersGeneral['id'], best_position])

# def run_pso(parametersGeneral, parametersPso):
#     pso = ps.PSO(paper_size=parametersGeneral["paper_size"],
#             image_sizes=parametersGeneral["image_sizes"],
#             population_size=parametersPso['population_size'],
#             desired_fitness=parametersGeneral["desired_fitness"],
#             iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"]/parametersPso['population_size'],
#             w=parametersPso['w'], c1=parametersPso['c1'], c2=parametersPso['c2'])

#     best_position = pso.run()

#     # Open CSV file in append mode
#     with open("testResultsPso.csv", 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([parametersGeneral['id'],
#                          pso.gbest_fitness,
#                          pso.population_size*pso.iterations,
#                          pso.iterations])
        
#     with open("testSolutionsPso.csv", 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([parametersGeneral['id'], best_position])

# def find_best_parameters(algorithm):
#     best_parameters = None
#     best_performance = float('inf')

#     if algorithm == 'hso':
#         HM_size_range = range(10, 210, 10)
#         memory_consideration_rate_range = np.arange(0.1, 1.1, 0.1)
#         pitch_adjustment_rate_range = np.arange(0.1, 1.1, 0.1)
#         pitch_bandwidth_range = np.arange(0.1, 1.1, 0.1)

#         for HM_size in HM_size_range:
#             for memory_consideration_rate in memory_consideration_rate_range:
#                 for pitch_adjustment_rate in pitch_adjustment_rate_range:
#                     for pitch_bandwidth in pitch_bandwidth_range:
#                         params = {
#                             'HM_size': HM_size,
#                             'memory_consideration_rate': memory_consideration_rate,
#                             'pitch_adjustment_rate': pitch_adjustment_rate,
#                             'pitch_bandwidth': pitch_bandwidth,
#                         }
#                         # Format each floating-point number in the dictionary
#                         formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in params.items()}
                        
#                         print(f"\rCurrent parameters: {formatted_params}                           ",
#                             end='', flush=True)
#                         fitness, iterations = run_hso(params)
#                         if fitness == 0:
#                             performance = iterations
#                         else:
#                             performance = fitness + iterations * 1000

#                         if performance < best_performance:
#                             best_performance = performance
#                             best_parameters = params

def find_best_parameters_hso(parametersGeneral):
    parameter_sets = [
        {'HM_size': 100, 'memory_consideration_rate': 0.8, 'pitch_adjustment_rate': 0.15, 'pitch_bandwidth': 0.15},
        {'HM_size': 100, 'memory_consideration_rate': 0.8, 'pitch_adjustment_rate': 0.15, 'pitch_bandwidth': 0.20},
        {'HM_size': 100, 'memory_consideration_rate': 0.8, 'pitch_adjustment_rate': 0.15, 'pitch_bandwidth': 0.25},
        {'HM_size': 100, 'memory_consideration_rate': 0.8, 'pitch_adjustment_rate': 0.15, 'pitch_bandwidth': 0.35},
        {'HM_size': 5, 'memory_consideration_rate': 0.9, 'pitch_adjustment_rate': 0.33, 'pitch_bandwidth': 0.01},
        {'HM_size': 50, 'memory_consideration_rate': 0.95, 'pitch_adjustment_rate': 0.1, 'pitch_bandwidth': 100},
    ]

    for parametersHso in parameter_sets:
        # Format each floating-point number in the dictionary
        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in parametersHso.items()}
        
        print(f"Current parameters: {formatted_params}", end='\r')
        
        hso = hs.HarmonySearch(paper_size=parametersGeneral["paper_size"],
                               image_sizes=parametersGeneral["image_sizes"],
                               dimensions=parametersGeneral["dimensions"],
                               iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"],
                               desired_fitness=parametersGeneral["desired_fitness"], 
                               HM_size=parametersHso['HM_size'],
                               memory_consideration_rate=parametersHso['memory_consideration_rate'],
                               pitch_adjustment_rate=parametersHso['pitch_adjustment_rate'],
                               pitch_bandwidth=parametersHso['pitch_bandwidth'])
        
        hso.run()
        
        with open("parameterSearchResultsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             hso.HM_size,
                             hso.hmcr, hso.par, hso.pb,
                             hso.best_fitness,
                             hso.iterations,
                             hso.iterations+hso.HM_size])
                        
        with open("parameterSearchSolutionsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], parametersGeneral["image_sizes"]])

def find_best_parameters_pso(parametersGeneral):
    parameter_sets = [
        {'population_size': 200, 'w': 0.05, 'c1': 5, 'c2': 2},
        {'population_size': 60, 'w': 0.3, 'c1': 0.8, 'c2': 0.8},
        {'population_size': 30, 'w': 0.8, 'c1': 2.0, 'c2': 2.0},
        {'population_size': 100, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'population_size': 50, 'w': 0.4, 'c1': 1.0, 'c2': 2.0},
        {'population_size': 100, 'w': 1.0, 'c1': 0.1, 'c2': 0.9},
    ]

    for parametersPso in parameter_sets:
        # Format each floating-point number in the dictionary
        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in parametersPso.items()}
        
        #sys.stdout.write('\033[K')
        print(f"Current parameters: {formatted_params}", end='\r')
        
        pso = ps.PSO(paper_size=parametersGeneral["paper_size"],
                     image_sizes=parametersGeneral["image_sizes"],
                     population_size=parametersPso["population_size"],
                     desired_fitness=parametersGeneral["desired_fitness"],
                     iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"]/parametersPso['population_size'],
                     w=parametersPso['w'], c1=parametersPso['c1'], c2=parametersPso['c2'])
        
        pso.run()
        
        with open("parameterSearchResultsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             pso.population_size,
                             pso.w, pso.c1, pso.c2,
                             pso.gbest_fitness,
                             pso.iterations,
                             pso.population_size * pso.iterations])
                        
        with open("parameterSearchSolutionsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], parametersGeneral["image_sizes"]])

# Example usage
if __name__ == "__main__":
    with open("parameterTestResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["HM size", "memory_consideration_rate", "pitch_adjustment_rate", "fitness"])

    with open("parameterSearchResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id","population_size", "w", "c1", "c2", "fitness", "iterations", "particles"])

    with open("parameterSearchInputs.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "image_sizes"])
    
##########################################################################################
        
    paramsGeneral = {
        'id': 1,
        'N': None,
        'setN': None,
        'runN': None,
        'paper_size': (100, 100),
        'image_sizes': None,
        'individuals_without_improvement_limit': 4000,
        'desired_fitness': 0,
    }

##########################################################################################
    print(f"Run id:{paramsGeneral["id"]}")

    max_image_count = 2
    image_set_generations = 1
    image_set_runs = 1


    for N in range(2, max_image_count+1):
        print(f"N:{N}")
        for setN in range(image_set_generations):
            image_sizes = imageCut.generate_image_sizes(N, paramsGeneral["paper_size"])
            with open("parameterSearchSolutionsPso.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([paramsGeneral['id'], image_sizes])
            for runN in range(image_set_runs):

                #sys.stdout.write('\033[K')
                print(f"Run id:{paramsGeneral["id"]}")
                dimensions = 3 * len(image_sizes)
                paramsGeneral["image_sizes"] = image_sizes
                paramsGeneral["dimensions"] = dimensions
                paramsGeneral["N"] = N
                paramsGeneral["setN"] = setN
                paramsGeneral["runN"] = runN
                
                find_best_parameters_pso(paramsGeneral)
                find_best_parameters_pso(paramsGeneral)
                
                paramsGeneral["id"] += 1

    print(f"\rTests done!",end='', flush=True)
    print("\n")
