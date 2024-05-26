import numpy as np
import sys
class PSO:
    def __init__(self, paper_size, image_sizes,
                 population_size=100, desired_fitness=0, 
                 iterations_without_improvement_limit=1000,
                 w=0.9, c1=2, c2=2, alpha=0.0001):
        
        self.paper_width, self.paper_height = paper_size
        self.image_sizes = np.array(image_sizes)
        self.N = len(image_sizes)

        self.population_size = population_size
        self.desired_fitness = desired_fitness
        self.iterations_without_improvement_limit = iterations_without_improvement_limit
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha

        self.sum_x_lengths = np.sum(self.image_sizes[:, 0])
        self.sum_y_lengths = np.sum(self.image_sizes[:, 1])
        
        self.max_scale = max(self.paper_height,self.paper_width)/min(val for sublist in self.image_sizes for val in sublist)
        self.min_scale = min(self.paper_height/self.sum_y_lengths,self.paper_width/self.sum_x_lengths)

        self.positions = np.empty((self.population_size, self.N, 3))
        self.positions[:, :, 0] = np.random.uniform(0, self.paper_width, (self.population_size, self.N))
        self.positions[:, :, 1] = np.random.uniform(0, self.paper_height, (self.population_size, self.N))
        #self.positions[:, :, 2] = np.ones((self.population_size, self.N))
        self.positions[:, :, 2] = np.random.uniform(self.min_scale, self.max_scale, (self.population_size, self.N))

        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.N, 3))

        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitnesses = np.full(self.population_size, float('inf'))

        self.gbest_position = np.zeros((1, self.N, 3))
        self.gbest_fitness = float('inf')

        self.iterations = 0
        self.iterations_without_improvement = 0
        self.reinitialization_N = 0

        self.k_neighbors = 10
        self.neighbor_indices = self.initialize_neighbors()

        self.max_resizing_deviation = 0
        for size in image_sizes:
            deviation_per_size = round(abs(self.max_scale - self.min_scale) / (1 / size[0]))
            deviation_per_size = round(abs(self.max_scale - self.min_scale) / (1 / size[1]))
            self.max_resizing_deviation += deviation_per_size
        
        self.total_area = paper_height * paper_width


    def compute_fitness_vectorized_np(self, scaling_penalty_factor=20, boundary_penalty_factor=10, 
                                      overlap_penalty_factor=10, uncovered_area_penalty_factor=1):
        
        out_of_bound_penalties = np.zeros(self.population_size)
        overlapping_area_penalties = np.zeros(self.population_size)
        uncovered_area_penalties = np.full(self.population_size, self.paper_width * self.paper_height, dtype=np.float64)
        self.biggest_possible_overlap = np.zeros(self.population_size)

        def get_image_data(image_idx):
            X = self.positions[:, image_idx:image_idx+1, 0:1]
            Y = self.positions[:, image_idx:image_idx+1, 1:2]
            S = self.positions[:, image_idx:image_idx+1, 2:3]
            
            W, H = self.image_sizes[image_idx:image_idx+1, 0:1]*S, self.image_sizes[image_idx:image_idx+1, 1:2]*S 

            X1 = X
            X2 = X + W
            Y1 = Y
            Y2 = Y + H
            W, H, X1, X2, Y1, Y2 = W.round(), H.round(), X1.round(), X2.round(), Y1.round(), Y2.round()
            return W.flatten(), H.flatten(), X1.flatten(), X2.flatten(), Y1.flatten(), Y2.flatten()
        
        def calculate_scaling_penalties():
            # Calculate the average scale for each particle
            average_scales = np.mean(self.positions[:, :, 2], axis=1, keepdims=True)
            
            # Calculate the squared differences from the average and sum these differences for each particle
            scale_differences = self.positions[:, :, 2] - average_scales
            significant = scale_differences > (self.positions[:, :, 2] / np.maximum(self.paper_height, self.paper_width))
            squared_differences = np.square(scale_differences)
            filtered_squared_differences = squared_differences * significant

            # Step 5: Calculate the scaling penalties by summing the filtered squared differences for each particle
            scaling_penalties = np.sum(filtered_squared_differences, axis=1)

            return scaling_penalties
        
        def calculate_out_of_bound_penalties_for_image(image_idx):
            W, H, X1, X2, Y1, Y2 = get_image_data(image_idx)

            # out_x = np.maximum(0, np.maximum(X1 - self.paper_width, 0) + np.maximum(0 - X2, 0))
            # out_y = np.maximum(0, np.maximum(Y1 - self.paper_height, 0) + np.maximum(0 - Y2, 0))

            # return out_x + out_y
            # Calculate the area of the rectangle and the inbound area
            area = np.maximum(W * H, 0.0)
            inbound = np.maximum(self.paper_width - np.maximum(X1, 0.0) - np.maximum(self.paper_width - np.maximum(X2, 0.0), 0.0), 0.0) * np.maximum(self.paper_height - np.maximum(Y1, 0.0) - np.maximum(self.paper_height -np.maximum(Y2, 0.0), 0.0), 0.0)


            # Calculate the out-of-bound penalty
            out_of_bound_penalty = np.maximum(area - inbound, 0.0)
            
            return out_of_bound_penalty

        # def calculate_out_of_bound_penalties_for_image(image_idx):
        #     W, H, X1, X2, Y1, Y2 = get_image_data(image_idx)

        #     # Total image area for each particle
        #     area = W * H

        #     # Clamping values to ensure they are within the bounds of the paper
        #     clamped_x1 = np.maximum(X1, 0)
        #     clamped_y1 = np.maximum(Y1, 0)
        #     clamped_x2 = np.minimum(X2, self.paper_width)
        #     clamped_y2 = np.minimum(Y2, self.paper_height)

        #     # Calculate the width and height of the in-bounds area
        #     # Ensuring positive or zero width and height by taking max with 0
        #     inbound_width = np.maximum(clamped_x2 - clamped_x1, 0)
        #     inbound_height = np.maximum(clamped_y2 - clamped_y1, 0)
        #     inbound_area = inbound_width * inbound_height

        #     # Out-of-bound area calculation
        #     out_of_bound_penalty = np.maximum(0, area - inbound_area)

        #     return out_of_bound_penalty

        
        def calculate_area_for_image(img_idx):
            W, H, X1, X2, Y1, Y2 = get_image_data(img_idx)
            
            # Calculate the area of the rectangle and the inbound area
            inbound = np.maximum(self.paper_width - np.maximum(X1, 0.0) - np.maximum(self.paper_width - np.maximum(X2, 0.0), 0.0), 0.0) * \
                    np.maximum(self.paper_height - np.maximum(Y1, 0.0) - np.maximum(self.paper_height - np.maximum(Y2, 0.0), 0.0), 0.0)

            return inbound
        
        # def calculate_overlap(image_idx_A, image_idx_B):
        #     _, _, XA1, XA2, YA1, YA2 = get_image_data(image_idx_A)
        #     WB, HB, XB1, XB2, YB1, YB2 = get_image_data(image_idx_B)

        #     # horizontal_overlap = np.maximum(0, np.minimum(XA2, XB2) - np.maximum(XA1, XB1))
        #     # vertical_overlap = np.maximum(0, np.minimum(YA2, YB2) - np.maximum(YA1, YB1))
        #     # overlap_area = horizontal_overlap * vertical_overlap

        #     # return overlap_area

        #     # Jei XA1 < XB1, tai ta rei6kinio dalis turi but 0, o ne WB
        #     np.where(XA1 < XB1, 0.0, np.maximum(WB - (XA1 - XB1), 0.0))

        #     overlap = np.maximum(np.where(XA1 < XB1, 0.0, np.maximum(WB - (XA1 - XB1), 0.0)) + 
        #                         np.where(XB2 < XA2, 0.0, np.maximum(WB - (XB2 - XA2), 0.0)), 0.0) * \
        #             np.maximum(np.where(YA1 < YB1, 0.0, np.maximum(HB - (YA1 - YB1), 0.0)) + 
        #                         np.where(YB2 < YA2, 0.0, np.maximum(HB - (YB2 - YA2), 0.0)), 0.0)

        #     overlap_penalty = np.maximum(overlap, 0.0)

        #     return overlap_penalty


        def calculate_overlap(image_idx_A, image_idx_B):
            WA, HA, XA1, XA2, YA1, YA2 = get_image_data(image_idx_A)
            WB, HB, XB1, XB2, YB1, YB2 = get_image_data(image_idx_B)

            # Calculate horizontal and vertical overlaps
            horizontal_overlap = np.maximum(0, np.minimum(XA2, XB2) - np.maximum(XA1, XB1))
            vertical_overlap = np.maximum(0, np.minimum(YA2, YB2) - np.maximum(YA1, YB1))

            # The area of overlap is simply the product of the horizontal and vertical overlaps
            overlap_area = horizontal_overlap * vertical_overlap
            self.biggest_possible_overlap += np.minimum(WA,WB) * np.minimum(HA,HB)
            return overlap_area


        for i in range(self.N):
            #print(out_of_bound_penalties.shape)
            out_of_bound_penalties += calculate_out_of_bound_penalties_for_image(i)
            for j in range(i+1, self.N):
                overlapping_area_penalties += calculate_overlap(i, j)
            # Reiktų pridėt overlapping area penalty tik tom vietom kurios yra inbound ??
            uncovered_area_penalties -= calculate_area_for_image(i) 

        uncovered_area_penalties = np.maximum(uncovered_area_penalties, 0.0)
        scaling_penalties = calculate_scaling_penalties()

        # Normalizing penalty weights
        scaling_penalties = scaling_penalties / self.max_resizing_deviation
        out_of_bound_penalties = out_of_bound_penalties / ( self.total_area * len(image_sizes))
        uncovered_area_penalties = uncovered_area_penalties / self.total_area
        overlapping_area_penalties = overlapping_area_penalties / self.biggest_possible_overlap

        fitness = scaling_penalties * scaling_penalty_factor + out_of_bound_penalties * boundary_penalty_factor +\
            uncovered_area_penalties * uncovered_area_penalty_factor + overlapping_area_penalties * overlap_penalty_factor
        return fitness
    
    # def update_velocity(self):
    #     r1 = np.random.uniform(0, 1, (self.population_size, self.N, 3))
    #     r2 = np.random.uniform(0, 1, (self.population_size, self.N, 3))
    #     cognitive_velocities = self.c1 * r1 * (self.pbest_positions - self.positions)
    #     social_velocities = self.c2 * r2 * (self.gbest_position - self.positions)
    #     self.velocities = self.w * self.velocities + cognitive_velocities + social_velocities

    def update_velocity(self):
        r1 = np.random.uniform(0, 1, (self.population_size, self.N, 3))
        r2 = np.random.uniform(0, 1, (self.population_size, self.N, 3))

        # Initialize local best positions array to match the shape of positions
        local_best_positions = np.zeros_like(self.positions)
        
        # Iterate over each particle to determine local best position
        for i in range(self.population_size):
            # Get indices of the neighbors
            neighbors = self.neighbor_indices[i]
            
            # Find the best fitness among the neighbors
            best_neighbor_idx = neighbors[np.argmin(self.pbest_fitnesses[neighbors])]
            
            # Store the position corresponding to the best neighbor fitness
            local_best_positions[i] = self.pbest_positions[best_neighbor_idx]

        cognitive_velocities = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_velocities = self.c2 * r2 * (local_best_positions - self.positions)
        self.velocities = self.w * self.velocities + cognitive_velocities + social_velocities


    def initialize_neighbors(self):
        neighbors = np.zeros((self.population_size, 2 * self.k_neighbors), dtype=int)
        for i in range(self.population_size):
            for j in range(1, self.k_neighbors+1):
                left = (i - j) % self.population_size
                right = (i + j) % self.population_size
                neighbors[i, j-1] = left
                neighbors[i, self.k_neighbors+j-1] = right
        return neighbors
    
    def update_position(self):
        self.positions += self.velocities
        min_scale_threshold = 0.0001
        # self.positions[:, :, 2] = np.maximum(self.positions[:, :, 2], min_scale_threshold)
    
    def run(self):
        while self.gbest_fitness > self.desired_fitness:
            if self.iterations_without_improvement > self.iterations_without_improvement_limit:
                # print(f"Stopping early after {self.iterations} iterations")
                # print(f"Best fitness: {self.gbest_fitness}")
                # print(f"Best position: {self.gbest_position}")
                #print(f"Stopping early after {self.iterations} iterations. Best fitness: {self.gbest_fitness}")
                break
            self.update_velocity()
            self.update_position()
            # self.positions[0] = [[0,0,1],[0,20,1],[0,40,1],[0,60,1]]
            self.fitnesses = self.compute_fitness_vectorized_np()

            better_idx = self.fitnesses < self.pbest_fitnesses
            
            # Update personal bests using proper indexing
            self.pbest_positions[better_idx] = self.positions[better_idx]
            self.pbest_fitnesses[better_idx] = self.fitnesses[better_idx]

            best_fitness_idx = np.argmin(self.fitnesses)
            if self.fitnesses[best_fitness_idx] < self.gbest_fitness:
                #sys.stdout.write('\033[K')
                print(f"New best fitness: {self.fitnesses[best_fitness_idx]}", end='\r')
                self.gbest_fitness = self.fitnesses[best_fitness_idx]
                self.gbest_position = self.positions[best_fitness_idx].copy()
                self.iterations_without_improvement = 0
                self.reinitialization_N = 0
            else:
                self.iterations_without_improvement += 1

            # Check and reinitialize velocities based on mean velocity magnitude
            mean_velocity = np.mean(np.linalg.norm(self.velocities, axis=1))
            #print(f"mean velocity {mean_velocity}", end='\r')
            if ((mean_velocity < self.alpha and self.gbest_fitness > 200) or ((self.iterations_without_improvement > self.iterations_without_improvement_limit/2) ))and self.reinitialization_N < 20 :
                self.reinitialization_N += 1
                # if self.iterations_without_improvement > self.iterations_without_improvement_limit/4:
                #     sys.stdout.write('\033[K')
                #     print(f"Reinitializing velocities: iterations without improvement: {self.iterations_without_improvement} \
                #           mean velocity: {mean_velocity}", end='\r')
                # print(f"Reinitializing velocities: mean velocity {mean_velocity} is less than  alpha {self.alpha}", end='\r')
                self.positions = np.empty((self.population_size, self.N, 3))
                self.positions[:, :, 0] = np.random.uniform(0, self.paper_width, (self.population_size, self.N))
                self.positions[:, :, 1] = np.random.uniform(0, self.paper_height, (self.population_size, self.N))
                #self.positions[:, :, 2] = np.ones((self.population_size, self.N))
                self.positions[:, :, 2] = np.random.uniform(self.min_scale, self.max_scale, (self.population_size, self.N))

                self.velocities = np.random.uniform(-1, 1, (self.population_size, self.N, 3))

                self.pbest_positions = np.copy(self.positions)
                self.pbest_fitnesses = np.full(self.population_size, float('inf'))

                # self.gbest_position = np.zeros((1, self.N, 3))
                # self.gbest_fitness = float('inf')

                self.iterations_without_improvement = 0

            self.iterations += 1

        return self.gbest_position


# Example usage:
if __name__ == "__main__":

    paper_width = 100
    paper_height = 80
    paper_size = (paper_width, paper_height)
    image_sizes = [[100,20],[100,20],[100,20],[100,20]]
    N = len(image_sizes)
    population_size = 100
    desired_fitness = 0
    iterations_without_improvement_limit = 2000
    w, c1, c2 = 0.729, 1.49455, 1.49455
    # w, c1, c2 = 0.7, 1, 2
    #w, c1, c2 = 0.3, 1.5, 1.5

    pso = PSO(paper_size=paper_size, 
              image_sizes=image_sizes, 
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
