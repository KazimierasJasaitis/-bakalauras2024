import hso as hs
import pso as ps
import csv
import sys

# Setup the constant parameters
paper_size = (100, 100)
image_sizes = [[50, 50], [50, 50], [50, 50], [50, 50]]
dimensions = 3 * len(image_sizes)
iterations_without_improvement_limit = 200
desired_fitness = 0

def run_hso(parameters):
    hso = hs.HarmonySearch(paper_size=paper_size, 
                        image_sizes=image_sizes, 
                        dimensions=dimensions, 
                        iterations_without_improvement_limit=iterations_without_improvement_limit*10, 
                        desired_fitness=desired_fitness, 
                        HM_size=parameters['HM_size'], 
                        memory_consideration_rate=parameters['memory_consideration_rate'], 
                        pitch_adjustment_rate=parameters['pitch_adjustment_rate'],
                        pitch_bandwidth=parameters['pitch_bandwidth'])
    
    best_position = hso.run()
    # Open CSV file in append mode
    with open("parameterTestResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parameters['HM_size'], 
                         parameters['memory_consideration_rate'], 
                         parameters['pitch_adjustment_rate'], 
                         hso.best_fitness,
                         hso.iterations])

    return hso.best_fitness, hso.iterations

def run_pso(parameters):

    pso = ps.PSO(paper_size=paper_size, 
              image_sizes=image_sizes, 
              dimensions=dimensions,
              population_size=parameters['population_size'], 
              desired_fitness=desired_fitness, 
              iterations_without_improvement_limit=(iterations_without_improvement_limit*10)/parameters['population_size'],
              w=parameters['w'], c1=parameters['c1'], c2=parameters['c2'])

    best_position = pso.run()

    # Open CSV file in append mode
    with open("parameterTestResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parameters['population_size'], parameters['w'], parameters['c1'], parameters['c2'], pso.gbest_fitness, pso.iterations, pso.population_size*pso.iterations])

    # Return the best fitness
    return pso.gbest_fitness, pso.iterations*pso.population_size



def find_best_parameters(algorithm):
    best_parameters = None
    best_performance = float('inf')

    # Define bounds and steps for each parameter
    if algorithm == 'hso':
        HM_size_range = range(10, 210, 10)
        memory_consideration_rate_range = np.arange(0.1, 1.1, 0.1)
        pitch_adjustment_rate_range = np.arange(0.1, 1.1, 0.1)
        pitch_bandwidth_range = np.arange(0.1, 1.1, 0.1)

        for HM_size in HM_size_range:
            for memory_consideration_rate in memory_consideration_rate_range:
                for pitch_adjustment_rate in pitch_adjustment_rate_range:
                    for pitch_bandwidth in pitch_bandwidth_range:
                        params = {
                            'HM_size': HM_size,
                            'memory_consideration_rate': memory_consideration_rate,
                            'pitch_adjustment_rate': pitch_adjustment_rate,
                            'pitch_bandwidth': pitch_bandwidth,
                        }
                        # Format each floating-point number in the dictionary
                        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in params.items()}
                        
                        print(f"\rCurrent parameters: {formatted_params}                           ",
                            end='', flush=True)
                        fitness, iterations = run_hso(params)
                        if fitness == 0:
                            performance = iterations
                        else:
                            performance = fitness + iterations * 1000

                        if performance < best_performance:
                            best_performance = performance
                            best_parameters = params

    elif algorithm == 'pso':
        population_size_range = range(10, 210, 10)
        w_range = np.arange(0.1, 1.1, 0.1)
        c1_range = np.arange(0.1, 2.1, 0.2)
        c2_range = np.arange(0.1, 2.1, 0.2)


        for population_size in population_size_range:
            for w in w_range:
                for c1 in c1_range:
                    for c2 in c2_range:
                        params = {
                            'population_size': population_size,
                            'w': w,
                            'c1': c1,
                            'c2': c2,
                        }
                        # Format each floating-point number in the dictionary
                        formatted_params = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in params.items()}
                        
                        print(f"\rCurrent parameters: {formatted_params}                   ",
                               end='', flush=True)
                        fitness, iterations = run_pso(params)
                        if fitness == 0:
                            performance = iterations
                        else:
                            performance = fitness + iterations * 1000

                        if performance < best_performance:
                            best_performance = performance
                            best_parameters = params

    return best_parameters, best_performance

# Example usage
if __name__ == "__main__":
    import numpy as np

    with open("parameterTestResultsHso.csv", 'a', newline='') as file:
        # Open CSV file to write header string
        writer = csv.writer(file)
        writer.writerow(["HM size", "memory_consideration_rate", "pitch_adjustment_rate", "fitness"])

    best_hso_params, best_hso_perf = find_best_parameters('hso')
    print(f"\rBest HSO Parameters: {best_hso_params} Performance: {best_hso_perf}            ",end='', flush=True)
    print("\n")

    with open("parameterTestResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["population_size", "w", "c1", "c2", "fitness", "iterations", "particles"])
    
    best_pso_params, best_pso_perf = find_best_parameters('pso')
    print(f"\rBest PSO Parameters: {best_pso_params} Performance: {best_pso_perf}            ",end='', flush=True)


