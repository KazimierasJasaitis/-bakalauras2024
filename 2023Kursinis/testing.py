import hso as hs
import pso as ps
import imageCut
import csv
import numpy as np

def run_hso(parametersGeneral, parametersHso):
    hso = hs.HarmonySearch(paper_size=parametersGeneral["paper_size"],
                        image_sizes=parametersGeneral["image_sizes"],
                        dimensions=parametersGeneral["dimensions"],
                        iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"],
                        desired_fitness=parametersGeneral["desired_fitness"], 
                        HM_size=parametersHso['HM_size'],
                        memory_consideration_rate=parametersHso['memory_consideration_rate'],
                        pitch_adjustment_rate=parametersHso['pitch_adjustment_rate'],
                        pitch_bandwidth=parametersHso['pitch_bandwidth'])

    best_position = hso.run()

    with open("testResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parametersGeneral['id'], hso.best_fitness, hso.iterations])
    
    with open("testSolutionsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parametersGeneral['id'], best_position])

def run_pso(parametersGeneral, parametersPso):
    pso = ps.PSO(paper_size=parametersGeneral["paper_size"],
            image_sizes=parametersGeneral["image_sizes"],
            dimensions=parametersGeneral["dimensions"],
            population_size=parametersPso['population_size'],
            desired_fitness=parametersGeneral["desired_fitness"],
            iterations_without_improvement_limit=parametersGeneral["individuals_without_improvement_limit"],
            w=parametersPso['w'], c1=parametersPso['c1'], c2=parametersPso['c2'])

    best_position = pso.run()

    # Open CSV file in append mode
    with open("testResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parametersGeneral['id'],
                         pso.gbest_fitness,
                         pso.population_size*pso.iterations,
                         pso.iterations])
        
    with open("testSolutionsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parametersGeneral['id'], best_position])


# Example usage
if __name__ == "__main__":

    with open("/TestResults/testResultsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "fitness", "iterations"])

    with open("testResultsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "fitness", "iterations"])

    with open("testSolutionsHso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "best_position"])

    with open("testSolutionsPso.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "best_position"])

##########################################################################################
        
    paramsGeneral = {
        'id': 1,
        'N': None,
        'setN': None,
        'runN': None,
        'paper_size': (100, 100),
        'image_sizes': None,
        'dimensions': None,
        'individuals_without_improvement_limit': 2000,
        'desired_fitness': 0,
    }
    paramsHso = {
        'HM_size': 100,
        'memory_consideration_rate': 1,
        'pitch_adjustment_rate': 0.3,
        'pitch_bandwidth': 0.6,
    }
    paramsPso = {
        'population_size': 110,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
    }

##########################################################################################
    print(f"\rRun id:{paramsGeneral["id"]}",end='', flush=True)

    max_image_count = 3
    image_set_generations = 5
    image_set_runs = 5


    for N in range(2, max_image_count+1):
        for setN in range(image_set_generations):
            image_sizes = imageCut.generate_image_sizes(N, paramsGeneral["paper_size"])
            for runN in range(image_set_runs):
                dimensions = 3 * len(image_sizes)
                paramsGeneral["image_sizes"] = image_sizes
                paramsGeneral["dimensions"] = dimensions
                paramsGeneral["N"] = N
                paramsGeneral["setN"] = setN
                paramsGeneral["runN"] = runN
                
                run_hso(paramsGeneral, paramsHso)
                run_pso(paramsGeneral, paramsPso)
                
                paramsGeneral["id"] += 1
                print(f"\rRun id:{paramsGeneral["id"]}",end='', flush=True)

    print(f"\rTests done!",end='', flush=True)
    print("\n")




