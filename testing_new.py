import hso as hs
import pso_vectorized_new as ps
import imageCut
import csv
import numpy as np
import sys

def run_hso(parametersGeneral):
    parameter_sets = [
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
        
        with open("TestResults/testResultsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             parametersGeneral['N'],
                             parametersHso['set'],
                             hso.HM_size,
                             hso.memory_consideration_rate, hso.pitch_adjustment_rate, hso.pitch_bandwidth,
                             hso.best_fitness,
                             hso.iterations,
                             hso.iterations+hso.HM_size])
                        
        with open("TestResults/testSolutionsHso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], best_positions])

def run_pso(parametersGeneral):
    parameter_sets = [
        {'set': 9, 'population_size': 50, 'w': 0.4, 'c1': 1.0, 'c2': 2.0},
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
        
        with open("TestResults/testResultsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'],
                             parametersGeneral['N'],
                             parametersPso['set'],
                             pso.population_size,
                             pso.w, pso.c1, pso.c2,
                             pso.gbest_fitness,
                             pso.iterations,
                             pso.population_size * pso.iterations])
                        
        with open("TestResults/testSolutionsPso.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parametersGeneral['id'], best_positions])

# Example usage
if __name__ == "__main__":
    with open("TestResults/testResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "N", "set", "HM size", "hmcr", "par", "pb", "fitness", "iterations", "particles"])

    with open("TestResults/testResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "N", "set", "population_size", "w", "c1", "c2", "fitness", "iterations", "particles"])

    with open("TestResults/testInputs.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "image_sizes"])

    with open("TestResults/testSolutionsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "positions"])

    with open("TestResults/testSolutionsHso.csv", 'a', newline='') as file:
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
        'individuals_without_improvement_limit': 20000,
        'desired_fitness': 0,
    }

##########################################################################################
    max_image_count = 5
    image_set_generations = 10
    image_set_runs = 20

    print("Starting tests...")
    sys.stdout.write('\n')
    for N in range(2, max_image_count+1):
        for setN in range(image_set_generations):
            image_sizes = imageCut.generate_image_sizes(N, paramsGeneral["paper_size"])
            with open("TestResults/testInputs.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([paramsGeneral['id'], image_sizes])
            for runN in range(image_set_runs):

                sys.stdout.write('\033[A')
                sys.stdout.flush()
                #sys.stdout.write('\n\r')
                sys.stdout.write(f'Run id:{paramsGeneral["id"]}, N:{N}, Image set:{setN+1}, Run:{runN}\r')
                sys.stdout.write('\n\r')
                sys.stdout.flush()

                paramsGeneral["image_sizes"] = image_sizes
                paramsGeneral["N"] = N
                paramsGeneral["setN"] = setN
                paramsGeneral["runN"] = runN
                
                try:
                    run_pso(paramsGeneral)
                    run_hso(paramsGeneral)
                except Exception as e:
                    print(f"An error occurred: {e}")

                
                paramsGeneral["id"] += 1
    sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear the line by writing enough spaces and return
    print("Tests done!")