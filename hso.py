import numpy as np
from PIL import Image, ImageDraw

class Harmony:
    def __init__(self, dimensions, image_sizes, paper_size):
        # Initialize harmony similar to Particle's initialization
        position_count = dimensions // 3
        x_coordinates = np.random.uniform(0, paper_size[0], position_count)
        y_coordinates = np.random.uniform(0, paper_size[1], position_count)
        scale_values = np.random.uniform(0.1, 1.0, position_count)  # Scale between 0.1 and 1 for simplicity

        # Merge coordinates and scale into one array
        self.position = np.empty(dimensions)
        self.position[0::3] = x_coordinates
        self.position[1::3] = y_coordinates
        self.position[2::3] = scale_values

        # Initial fitness
        self.fitness = float('inf')

    def compute_fitness(self, image_sizes, paper_size, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=5, uncovered_area_penalty_factor=5):
        paper_height, paper_width = paper_size
        total_area = paper_height * paper_width
        sum_image_areas = 0
        total_resizing_deviation = 0
        max_resizing_deviation = 0
        overlapping_area = 0
        boundary_penalty = 0
        overlapping_area_penalty = 0
        covered_area = 0
        covered_matrix = np.zeros(paper_size, dtype=bool)
        biggest_possible_overlap = 0

        positions = self.position.reshape(-1, 3)
        avg_scale = np.mean([scale for _, _, scale in positions])

        for i, (x, y, scale) in enumerate(positions):
            x=round(x)
            y=round(y)
            # Penalize negative or 0 scales
            if scale <= 0:
                fitness = float('inf')
                return fitness
            
            original_width, original_height = image_sizes[i]

            # Calculate the new dimensions of the image after resizing
            new_width = round(original_width * scale)
            new_height = round(original_height * scale)

            if new_width <= 0 or new_height <= 0: 
                fitness = float('inf') 
                return fitness 

            # Add to the sum of image areas
            image_area = new_width * new_height
            sum_image_areas += image_area

            # Check for overlaps with other images
            for j in range(i + 1, len(positions)):
                x2, y2, scale2 = positions[j]
                x2=round(x2)
                y2=round(y2)
                original_width2, original_height2 = image_sizes[j]
                new_width2 = round(original_width2 * scale2)
                new_height2 = round(original_height2 * scale2)
                if new_width2 <= 0 or new_height2 <= 0: 
                    fitness = float('inf') 
                    return fitness 

                overlap_height = min((y + new_height),y2 + new_height2)-max(y,y2)
                overlap_width = min((x+new_width),x2 + new_width2)-max(x,x2)

                overlapping_area += max(0,overlap_height * overlap_width)

                biggest_overlap_height = min(new_height,new_height2)
                biggest_overlap_width = min(new_width,new_width2)
                biggest_possible_overlap += biggest_overlap_height * biggest_overlap_width     

            # Check for out of boundary
            if (x + new_width > paper_height or y + new_height > paper_width or x < 0 or y < 0):
                in_bound_height = min((y+new_height),paper_height)-max(y,0)
                in_bound_width = min((x+new_width),paper_width)-max(x,0)
                # Calculate area inside the bounds
                in_bounds_area = in_bound_height * in_bound_width
                # Calculate total out-of-bound area
                out_of_bounds_area = (image_area) - in_bounds_area
                boundary_penalty += max(0,out_of_bounds_area)

            # Calculate total x and y lengths of all images
            sum_x_lengths = np.sum([size[0] for size in image_sizes])
            sum_y_lengths = np.sum([size[1] for size in image_sizes])

            # Calculate min and max scale values
            max_scale = max(paper_height,paper_width)/min(val for sublist in image_sizes for val in sublist)
            min_scale = min(paper_height/sum_y_lengths,paper_width/sum_x_lengths)
            
            # Biggest resizing deviation
            max_resizing_deviation += round(abs(max_scale - min_scale) / (1/original_width)) # For each pixel that is out of place from the average scale scenario
            max_resizing_deviation += round(abs(max_scale - min_scale) / (1/original_height)) # Same for width

            # Calculate the resizing deviation
            total_resizing_deviation += round(abs(avg_scale - scale) / (1/original_width)) # For each pixel that is out of place from the average scale scenario
            total_resizing_deviation += round(abs(avg_scale - scale) / (1/original_height)) # Same for width

            # Calculate uncovered area
            overlap = covered_matrix[y:y + new_height, x:x + new_width] # Check if the current image overlaps with already covered area
            covered_area += np.sum(~overlap) # Calculate the uncovered area for the current image
            covered_matrix[y:y + new_height, x:x + new_width] = True # Mark the newly covered area by the current image as True in the matrix

            uncovered_area = total_area - covered_area

        # Normalizing penalty weights
        total_resizing_deviation = total_resizing_deviation / max_resizing_deviation
        boundary_penalty = boundary_penalty / ( total_area * len(image_sizes))
        uncovered_area_penalty = uncovered_area / total_area
        overlapping_area_penalty = overlapping_area / biggest_possible_overlap
        # Compute the penalties
        fitness =   total_resizing_deviation * scaling_penalty_factor + \
                    boundary_penalty * boundary_penalty_factor + \
                    uncovered_area_penalty * uncovered_area_penalty_factor + \
                    overlapping_area_penalty * overlap_penalty_factor
        #print("area: ",sum_image_areas,", overlapping: ", overlapping_area, "boundary: ", boundary_penalty )
        #if fitness < 0: print(f"Area: {sum_image_areas}, Overlapping: {overlapping_area}, Boundary: {boundary_penalty}")
        #if fitness <= 0: print(f"1: {total_resizing_deviation}, 2: {boundary_penalty}, 3: {uncovered_area_penalty}, 4:{overlapping_area_penalty}") 

        return fitness

class HarmonySearch:
    def __init__(self, paper_size, image_sizes, dimensions, 
                 iterations_without_improvement_limit=float('inf'), 
                 desired_fitness=0, HM_size=50,
                 memory_consideration_rate=0.8, pitch_adjustment_rate=0.4, 
                 pitch_bandwidth=0.1):
        self.paper_size = paper_size
        self.image_sizes = image_sizes
        self.dimensions = dimensions
        self.iterations_without_improvement_limit = iterations_without_improvement_limit        
        self.desired_fitness = desired_fitness
        self.HM_size = HM_size
        self.memory_consideration_rate = memory_consideration_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.pitch_bandwidth = pitch_bandwidth

        self.best_harmony = None
        self.best_fitness = float('inf')
        self.iterations = 0
        self.iterations_without_improvement = 0

        # Set lower and upper bounds
        self.lower_bound = [0] * dimensions  # Example default value
        self.upper_bound = [max(paper_size)] * dimensions  # Example default value

        self.HM = [Harmony(dimensions, image_sizes, paper_size) for _ in range(HM_size)]

    def improvise_new_harmony(self):
        new_harmony = Harmony(self.dimensions, self.image_sizes, self.paper_size)
        for i in range(self.dimensions):
            if np.random.rand() < self.memory_consideration_rate:  # Harmony memory consideration
                new_harmony.position[i] = np.random.choice([h.position[i] for h in self.HM])
                if np.random.rand() < self.pitch_adjustment_rate:  # Pitch adjustment
                    new_harmony.position[i] += np.random.uniform(-1, 1) * self.pitch_bandwidth
            else:
                # Random selection for the dimension
                new_harmony.position[i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i])
        new_harmony.fitness = new_harmony.compute_fitness(self.image_sizes, self.paper_size)
        return new_harmony

    def update_harmony_memory(self, new_harmony):
        worst_harmony = max(self.HM, key=lambda h: h.fitness)
        if new_harmony.fitness < worst_harmony.fitness:
            self.HM.remove(worst_harmony)
            self.HM.append(new_harmony)

    def run(self):
        # Evaluate initial harmonies
        for harmony in self.HM:
            harmony.fitness = harmony.compute_fitness(self.image_sizes, self.paper_size)
            if harmony.fitness < self.best_fitness:
                self.best_fitness = harmony.fitness
                self.best_harmony = harmony

        # Main loop for Harmony Search
        while self.best_fitness > self.desired_fitness:
            if self.iterations_without_improvement >= self.iterations_without_improvement_limit:
                break
            self.iterations_without_improvement += 1
            self.iterations += 1

            # if (self.iterations % 1000) == 0:
            #     print(f"iterations: {self.iterations}; best fitness: {self.best_fitness}")

            new_harmony = self.improvise_new_harmony()
            new_fitness = new_harmony.compute_fitness(self.image_sizes, self.paper_size)
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_harmony = new_harmony
                self.iterations_without_improvement = 0
            self.update_harmony_memory(new_harmony)
        
        return self.best_harmony.position

# Example usage
if __name__ == "__main__":
    # Setup parameters
    paper_size = (100, 100)
    image_sizes = [[100, 50], [100, 50]]
    dimensions = 3 * len(image_sizes)
    iterations_without_improvement_limit = 1000
    desired_fitness = 0
    HM_size = 50
    memory_consideration_rate = 0.8
    pitch_adjustment_rate = 0.4
    pitch_bandwidth = 0.1

    hso = HarmonySearch(paper_size=paper_size, 
                        image_sizes=image_sizes, 
                        dimensions=dimensions, 
                        iterations_without_improvement_limit=iterations_without_improvement_limit, 
                        desired_fitness=desired_fitness, 
                        HM_size=HM_size, 
                        memory_consideration_rate=memory_consideration_rate, 
                        pitch_adjustment_rate=pitch_adjustment_rate, 
                        pitch_bandwidth=pitch_bandwidth)
    
    best_position = hso.run()

    # Print each image's position and scale factor
    best_position_2d = best_position.reshape(-1, 3)
    for i, (x, y, scale) in enumerate(best_position_2d):
        print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {round(scale, 2)}")

    print(hso.iterations)


    # Visualization (same as in PSO)
    img = Image.new('RGB', paper_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for (x, y, scale), size in zip(best_position_2d, image_sizes):
        rect = [x, y, x + size[0] * scale, y + size[1] * scale]
        draw.rectangle(rect, outline="blue", width=1)
    img.show()
