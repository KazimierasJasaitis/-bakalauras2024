import numpy as np

class Particle:
    def __init__(self, dimensions, min_scale, max_scale, paper_size):
        # Number of images
        position_count = dimensions // 3
        x_coordinates = np.empty(position_count)
        y_coordinates = np.empty(position_count)
        scale_values = np.empty(position_count)

        self.min_scale = min_scale
        self.max_scale = max_scale

        paper_height, paper_width = paper_size
        for i in range(position_count):
            # Randomly initialize x and y coordinates
            x_coordinates[i] = np.random.uniform(0, paper_width)
            y_coordinates[i] = np.random.uniform(0, paper_height)
            scale_values[i] = np.random.uniform(min_scale, max_scale)

        # Merge x_coordinates, y_coordinates, and scale_value into self.position
        self.position = np.empty(3 * position_count)
        self.position[0::3] = x_coordinates
        self.position[1::3] = y_coordinates
        self.position[2::3] = scale_values

        # Initialize velocity with random values between 0 and 1
        self.velocity = np.random.uniform(0, 1, dimensions)
        self.pbest_position = self.position
        self.pbest_fitness = float('inf')
        
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
            
            # Biggest resizing deviation
            max_resizing_deviation += round(abs(self.max_scale - self.min_scale) / (1/original_width)) # For each pixel that is out of place from the average scale scenario
            max_resizing_deviation += round(abs(self.max_scale - self.min_scale) / (1/original_height)) # Same for width

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
    
    def update_position(self):
        self.position = self.position + self.velocity

    def update_velocity(self, gbest_position, w, c1, c2):
        r1 = np.random.uniform(0, 1, len(self.position))
        r2 = np.random.uniform(0, 1, len(self.position))
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position)
        social_velocity = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity


class PSO:
    def __init__(self, paper_size, image_sizes, dimensions, 
                 iterations_without_improvement_limit=float('inf'),
                 desired_fitness=0, population_size=100,
                 w=0.9, c1=2, c2=2):
        self.paper_size = paper_size
        self.image_sizes = image_sizes
        self.dimensions = dimensions
        self.iterations_without_improvement_limit = iterations_without_improvement_limit
        self.desired_fitness = desired_fitness
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = float('inf')
        self.iterations = 0
        self.iterations_without_improvement = 0

        # Calculate total x and y lengths of all images
        self.sum_x_lengths = np.sum([size[0] for size in image_sizes])
        self.sum_y_lengths = np.sum([size[1] for size in image_sizes])

        paper_height, paper_width = paper_size
        # Calculate min and max scale values
        self.max_scale = max(paper_height,paper_width)/min(val for sublist in image_sizes for val in sublist)
        self.min_scale = min(paper_height/self.sum_y_lengths,paper_width/self.sum_x_lengths)
        self.population = [Particle(dimensions, self.min_scale, self.max_scale, self.paper_size) for _ in range(population_size)]

    def run(self):
        while self.gbest_fitness > self.desired_fitness:
            if self.iterations_without_improvement >= self.iterations_without_improvement_limit:
                break
            self.iterations_without_improvement += 1
            self.iterations += 1
            # print(f"\r{self.iterations}", end='', flush=True)
            for particle in self.population:
                fitness = particle.compute_fitness(self.image_sizes, self.paper_size)
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position
                    
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = particle.position
                        self.iterations_without_improvement = 0
                        
                        # print(self.gbest_fitness)
                        # best_position_2d = self.gbest_position.reshape(-1, 3)
                        # # Print each image's position and scale factor
                        # for i, (x, y, scale) in enumerate(best_position_2d):
                        #     print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {round(scale,2)}")
                        # print(pso.iterations)
                        # print("\n")
                        

            #print(self.population[2].velocity[11],self.population[2].pbest_fitness,self.population[2].position)
            for particle in self.population:
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position()
        return self.gbest_position



# Example usage:
if __name__ == "__main__":

    paper_width = 100
    paper_height = 150
    paper_size = (paper_width, paper_height)
    image_sizes = [[1000,500],[1000,500],[500,500],[500,500]]
    N = len(image_sizes)
    dimensions = 3 * N
    population_size = 50
    desired_fitness = 0
    iterations_without_improvement_limit = 1000
    w = 0.7
    c1 = 1
    c2 = 2

    pso = PSO(paper_size=paper_size, 
              image_sizes=image_sizes, 
              dimensions=dimensions,
              population_size=population_size, 
              desired_fitness=desired_fitness, 
              iterations_without_improvement_limit=iterations_without_improvement_limit,
              w=w, c1=c1, c2=c2)

    best_position = pso.run()
    print("\n")
    print(pso.gbest_fitness)
    best_position_2d = best_position.reshape(-1, 3)

    # Print each image's position and scale factor
    for i, (x, y, scale) in enumerate(best_position_2d):
        print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {round(scale,2)}")
    print(pso.iterations)


    from PIL import Image, ImageDraw

    # Create a blank canvas
    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw each rectangle
    for (x, y, scale), size in zip(best_position_2d, image_sizes):
        rect = [x, y, x + size[0] * scale, y + size[1] * scale]
        draw.rectangle(rect, outline ="blue", width=1)

    # Show the image
    img.show()
