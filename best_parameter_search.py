import hso as hs
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
#                         hmcr=parametersHso['hmcr'],
#                         par=parametersHso['par'],
#                         pb=parametersHso['pb'])

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
#         hmcr_range = np.arange(0.1, 1.1, 0.1)
#         par_range = np.arange(0.1, 1.1, 0.1)
#         pb_range = np.arange(0.1, 1.1, 0.1)

#         for HM_size in HM_size_range:
#             for hmcr in hmcr_range:
#                 for par in par_range:
#                     for pb in pb_range:
#                         params = {
#                             'HM_size': HM_size,
#                             'hmcr': hmcr,
#                             'par': par,
#                             'pb': pb,
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
        {'set': 1, 'HM_size': 100, 'hmcr': 0.8, 'par': 0.15, 'pb': 0.15},
        {'set': 2, 'HM_size': 100, 'hmcr': 0.8, 'par': 0.15, 'pb': 0.20},
        {'set': 3, 'HM_size': 100, 'hmcr': 0.8, 'par': 0.15, 'pb': 0.25},
        {'set': 4, 'HM_size': 100, 'hmcr': 0.8, 'par': 0.15, 'pb': 0.35},
        {'set': 5, 'HM_size': 5, 'hmcr': 0.9, 'par': 0.33, 'pb': 0.01},
        {'set': 6, 'HM_size': 50, 'hmcr': 0.95, 'par': 0.1, 'pb': 1.00},
        {'set': 7, 'HM_size': 50, 'hmcr': 0.95, 'par': 0.4, 'pb': 0.1},
    ]

    for parametersHso in parameter_sets:
        # Format each floating-point number in the dictionary
        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in parametersHso.items()}
        
        sys.stdout.write(f'Current HSO parameters: {formatted_params}             \r')
        sys.stdout.flush()

        dimensions = len(parametersGeneral["image_sizes"]) * 3
        hso = hs.HarmonySearch(paper_size=parametersGeneral["paper_size"],
                        image_sizes=image_sizes, 
                        dimensions=dimensions, 
                        iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"],
                        desired_fitness=parametersGeneral["desired_fitness"], 
                        HM_size=parametersHso['HM_size'], 
                        memory_consideration_rate=parametersHso['hmcr'],
                        pitch_adjustment_rate=parametersHso['par'],
                        pitch_bandwidth=parametersHso['pb'])
    
        best_positions = hso.run()        

        with open("TestResults/parameterSearchResultsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             parametersGeneral['N'],
                             parametersHso['set'],
                             hso.HM_size,
                             hso.memory_consideration_rate, hso.pitch_adjustment_rate, hso.pitch_bandwidth,
                             hso.best_fitness,
                             hso.iterations,
                             hso.iterations+hso.HM_size])
                        
        with open("TestResults/parameterSearchSolutionsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], best_positions])

def find_best_parameters_pso(parametersGeneral):
    parameter_sets = [
        {'set': 1, 'population_size': 200, 'w': 0.05, 'c1': 5, 'c2': 2},
        {'set': 2, 'population_size': 60, 'w': 0.3, 'c1': 0.8, 'c2': 0.8},
        {'set': 3, 'population_size': 30, 'w': 0.8, 'c1': 2.0, 'c2': 2.0},
        {'set': 4, 'population_size': 50, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'set': 5, 'population_size': 100, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'set': 6, 'population_size': 300, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'set': 7, 'population_size': 600, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'set': 8, 'population_size': 1000, 'w': 0.729, 'c1': 1.49455, 'c2': 1.49455},
        {'set': 9, 'population_size': 50, 'w': 0.4, 'c1': 1.0, 'c2': 2.0},
        {'set': 10, 'population_size': 100, 'w': 1.0, 'c1': 0.1, 'c2': 0.9},
    ]

    for parametersPso in parameter_sets:
        # Format each floating-point number in the dictionary
        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in parametersPso.items()}
        
        #sys.stdout.write('\033[K')
        #print(f"Current parameters: {formatted_params}", end='\r')
        sys.stdout.write(f'Current PSO parameters: {formatted_params}                  \r')
        sys.stdout.flush()
        
        pso = ps.PSO(paper_size=parametersGeneral["paper_size"],
                     image_sizes=parametersGeneral["image_sizes"],
                     population_size=parametersPso["population_size"],
                     desired_fitness=parametersGeneral["desired_fitness"],
                     iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"]/parametersPso['population_size'],
                     w=parametersPso['w'], c1=parametersPso['c1'], c2=parametersPso['c2'])
        
        best_positions = pso.run()
        
        with open("TestResults/parameterSearchResultsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             parametersGeneral['N'],
                             parametersPso['set'],
                             pso.population_size,
                             pso.w, pso.c1, pso.c2,
                             pso.gbest_fitness,
                             pso.iterations,
                             pso.population_size * pso.iterations])
                        
        with open("TestResults/parameterSearchSolutionsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], best_positions])

# Example usage
if __name__ == "__main__":
    with open("TestResults/parameterSearchResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "N", "set", "HM size", "hmcr", "par", "pb", "fitness", "iterations", "particles"])

    with open("TestResults/parameterSearchResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "N", "set", "population_size", "w", "c1", "c2", "fitness", "iterations", "particles"])

    with open("TestResults/parameterSearchInputs.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "image_sizes"])

    with open("TestResults/parameterSearchSolutionsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "positions"])

    with open("TestResults/parameterSearchSolutionsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "positions"])
    
##########################################################################################
        
    paramsGeneral = {
        'id': 1,
        'N': None,
        'setN': None,
        'runN': None,
        'paper_size': (100, 100),
        'image_sizes': None,
        'individuals_without_improvement_limit': 1000,
        'desired_fitness': 0,
    }

##########################################################################################
    max_image_count = 5
    image_set_generations = 5
    image_set_runs = 10

    print("Starting tests...")
    sys.stdout.write('\n')
    for N in range(2, max_image_count+1):
        for setN in range(image_set_generations):
            image_sizes = imageCut.generate_image_sizes(N, paramsGeneral["paper_size"])
            with open("TestResults/parameterSearchInputs.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([paramsGeneral['id'], image_sizes])
            for runN in range(image_set_runs):

                sys.stdout.write('\033[A')
                sys.stdout.flush()
                #sys.stdout.write('\n\r')
                sys.stdout.write(f'Run id:{paramsGeneral["id"]}, N:{N}, Set:{setN}, Run:{runN}\r')
                sys.stdout.write('\n\r')
                sys.stdout.flush()

                paramsGeneral["image_sizes"] = image_sizes
                paramsGeneral["N"] = N
                paramsGeneral["setN"] = setN
                paramsGeneral["runN"] = runN
                
                try:
                    # find_best_parameters_pso(paramsGeneral)
                    find_best_parameters_hso(paramsGeneral)
                except Exception as e:
                    print(f"An error occurred: {e}")

                
                paramsGeneral["id"] += 1
    sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear the line by writing enough spaces and return
    print("Tests done!")