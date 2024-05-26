import numpy as np

class PSO:
    def __init__(self, paper_size, image_sizes,
                 population_size=100, desired_fitness=0, 
                 iterations_without_improvement_limit=1000,
                 w=0.9, c1=2, c2=2):
        
        self.paper_width, self.paper_height = paper_size
        self.image_sizes = image_sizes
        self.N = len(image_sizes)

        self.population_size = population_size
        self.dimensions = dimensions
        self.desired_fitness = desired_fitness
        self.iterations_without_improvement_limit = iterations_without_improvement_limit
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Calculate total x and y lengths of all images
        self.sum_x_lengths = np.sum([size[0] for size in image_sizes])
        self.sum_y_lengths = np.sum([size[1] for size in image_sizes])
        
        self.max_scale = max(self.paper_height,self.paper_width)/min(val for sublist in image_sizes for val in sublist)
        self.min_scale = min(self.paper_height/self.sum_y_lengths,self.paper_width/self.sum_x_lengths)

        # Initialize positions: [x, y, scale] for each dimension set for each particle
        # Creating uniform distributions for each parameter across all particles
        self.positions = np.empty((self.population_size, self.N*3))
        self.positions[:, 0::3] = np.random.uniform(0, paper_size[1], (self.population_size, self.N))  # x-coordinates
        self.positions[:, 1::3] = np.random.uniform(0, paper_size[0], (self.population_size, self.N))  # y-coordinates
        self.positions[:, 2::3] = np.random.uniform(self.min_scale, self.max_scale, (self.population_size, self.N))  # scales

        self.velocities = np.random.uniform(-1, 1, (population_size, self.N*3))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitnesses = np.full(population_size, float('inf'))
        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = float('inf')

        self.iterations = 0
        self.iterations_without_improvement = 0
        self.image_sizes = image_sizes
        self.paper_size = paper_size

    def compute_fitness_vectorized_np(self, pos_fit, sizes_fit, paper_size, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=100, uncovered_area_penalty_factor=1):
        out_of_bound_penalty = 0
        overlapping_area_penalty = 0
        uncovered_area_penalty = paper_size[0] * paper_size[1]

        def calculate_out_of_bound_penalty_for_image(pos_fit, sizes_fit, paper_size, image_index):
            # Calculate column indices for the given image
            start_idx = image_index * 3
            X = pos_fit[:, start_idx:start_idx+1]
            Y = pos_fit[:, start_idx+1:start_idx+2]
            S = pos_fit[:, start_idx+2:start_idx+3]/100
            
            # Assuming sizes_fit has dimensions [batch_size, N, 2] and contains the original width and height for each image
            W, H = sizes_fit.reshape(-1, 5, 2)[:, image_index,0]*S, sizes_fit.reshape(-1, 5, 2)[:,image_index,1]*S
            
            paper_h, paper_w = paper_size
            
            X1 = X
            X2 = X + W
            Y1 = Y
            Y2 = Y + H
            
            # Calculate the area of the rectangle and the inbound area
            area = np.maximum(W * H, 0.0)
            inbound = np.maximum(paper_w - np.maximum(X1, 0.0) - np.maximum(paper_w - np.maximum(X2, 0.0), 0.0), 0.0) * np.maximum(paper_h - np.maximum(Y1, 0.0) - np.maximum(paper_h -np.maximum(Y2, 0.0), 0.0), 0.0)


            # Calculate the out-of-bound penalty
            out_of_bound_penalty = np.maximum(area - inbound, 0.0)

            negative_penalty = np.abs(np.minimum(X, 0.0)) + np.abs(np.minimum(Y, 0.0))+ np.abs(np.minimum(S, 0.0))*1000
            
            return out_of_bound_penalty + negative_penalty
        
        def calculate_area_for_image(pos_fit, sizes_fit, paper_size, image_index):
            # Calculate column indices for the given image
            start_idx = image_index * 3
            X = pos_fit[:, start_idx:start_idx+1]
            Y = pos_fit[:, start_idx+1:start_idx+2]
            S = pos_fit[:, start_idx+2:start_idx+3]/100
            
            # Assuming sizes_fit has dimensions [batch_size, N, 2] and contains the original width and height for each image
            W, H = sizes_fit.reshape(-1, 5, 2)[:, image_index,0]*S, sizes_fit.reshape(-1, 5, 2)[:,image_index,1]*S
            
            paper_h, paper_w = paper_size
            
            # print(f"Shapes - X: {X.shape}, Y: {Y.shape}, S: {S.shape}")
            # print(f"X: {X}, Y: {Y}, S: {S}")
            # print(f"Shapes - W: {W.shape}, H: {H.shape}")
            # print(f"W: {W}, H: {H}")
            # print(" ")
            X1 = X
            X2 = X + W
            Y1 = Y
            Y2 = Y + H
            
            # Calculate the area of the rectangle and the inbound area
            inbound = np.maximum(paper_w - np.maximum(X1, 0.0) - np.maximum(paper_w - np.maximum(X2, 0.0), 0.0), 0.0) * \
                    np.maximum(paper_h - np.maximum(Y1, 0.0) - np.maximum(paper_h - np.maximum(Y2, 0.0), 0.0), 0.0)
                        
            return inbound
        
        def calculate_overlap(pos_fit, sizes_fit, image_index_A, image_index_B):
            start_idx_A = image_index_A * 3
            start_idx_B = image_index_B * 3
            
            XA = pos_fit[:, start_idx_A:start_idx_A+1]
            XB = pos_fit[:, start_idx_B:start_idx_B+1]

            YA = pos_fit[:, start_idx_A+1:start_idx_A+2]
            YB = pos_fit[:, start_idx_B+1:start_idx_B+2]

            SA = pos_fit[:, start_idx_A+2:start_idx_A+3]/100
            SB = pos_fit[:, start_idx_B+2:start_idx_B+3]/100

            # Assuming sizes_fit has dimensions [batch_size, N, 2]
            WA, HA = sizes_fit.reshape(-1, 5, 2)[:, image_index_A,0]*SA, sizes_fit.reshape(-1, 5, 2)[:,image_index_A,1]*SA
            WB, HB = sizes_fit.reshape(-1, 5, 2)[:, image_index_B,0]*SB, sizes_fit.reshape(-1, 5, 2)[:,image_index_B,1]*SB
            
            XA1 = XA
            XB1 = XB

            XA2 = XA + WA
            XB2 = XB + WB

            YA1 = YA
            YB1 = YB

            YA2 = YA + HA
            YB2 = YB + HB

            # Jei XA1 < XB1, tai ta rei6kinio dalis turi but 0, o ne WB
            np.where(XA1 < XB1, 0.0, np.maximum(WB - (XA1 - XB1), 0.0))

            overlap = np.maximum(np.where(XA1 < XB1, 0.0, np.maximum(WB - (XA1 - XB1), 0.0)) + 
                                np.where(XB2 < XA2, 0.0, np.maximum(WB - (XB2 - XA2), 0.0)), 0.0) * \
                    np.maximum(np.where(YA1 < YB1, 0.0, np.maximum(HB - (YA1 - YB1), 0.0)) + 
                                np.where(YB2 < YA2, 0.0, np.maximum(HB - (YB2 - YA2), 0.0)), 0.0)
            # if np.any(overlap > 0):            
            #     print(f"Overlap: {overlap}, image_index_A: {image_index_A}, image_index_B: {image_index_B}")
            #     print(f"XA: {XA}, YA: {YA}, SA: {SA}")
            #     print(f"WA: {WA}, H: {HA}")
            #     print(f"XB: {XB}, YB: {YB}, SB: {SB}")
            #     print(f"WB: {WB}, HB: {HB}")
            #     print(" ")
            overlap_penalty = np.maximum(overlap, 0.0)

            return overlap_penalty
        
        for i in range(5):
            out_of_bound_penalty += calculate_out_of_bound_penalty_for_image(pos_fit, sizes_fit, paper_size, i)
            for j in range(i+1, 5):
                overlapping_area_penalty += calculate_overlap(pos_fit, sizes_fit, i, j)
                # print(overlapping_area_penalty)
                # print(i,j)
                # print(" ")
            # Reiktų pridėt overlapping area penalty tik tom vietom kurios yra inbound
            uncovered_area_penalty -= calculate_area_for_image(pos_fit, sizes_fit, paper_size, i) 
            
        uncovered_area_penalty = np.maximum(uncovered_area_penalty, 0.0)

        # print(f"Out of bound penalty: {out_of_bound_penalty}")
        # print(f"Overlapping area penalty: {overlapping_area_penalty}")
        # print(f"Uncovered area penalty: {uncovered_area_penalty}")
        # print(" ")
        fitness = out_of_bound_penalty * boundary_penalty_factor + uncovered_area_penalty * uncovered_area_penalty_factor +\
                overlapping_area_penalty * overlap_penalty_factor
        return fitness.flatten()
    

    def update_velocity(self):
        r1 = np.random.uniform(0, 1, (self.population_size, self.dimensions))
        r2 = np.random.uniform(0, 1, (self.population_size, self.dimensions))
        cognitive_velocities = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_velocities = self.c2 * r2 * (self.gbest_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive_velocities + social_velocities

    def update_position(self):
        self.positions += self.velocities
    
    def run(self):
        # Example loop for running the PSO
        while self.gbest_fitness > self.desired_fitness:
            self.update_velocity()
            self.update_position()
            fitnesses = self.compute_fitness_vectorized_np(self.positions, np.array(self.image_sizes), self.paper_size)

            # Ensure fitnesses is properly formatted and correctly sized
            if fitnesses.shape != (self.population_size,):
                raise ValueError(f"Expected fitnesses to be of shape ({self.population_size},), got {fitnesses.shape}")

            better_idx = fitnesses < self.pbest_fitnesses
            
            # Update personal bests using proper indexing
            self.pbest_positions[better_idx, :] = self.positions[better_idx, :]
            self.pbest_fitnesses[better_idx] = fitnesses[better_idx]

            best_fitness_idx = np.argmin(fitnesses)
            if fitnesses[best_fitness_idx] < self.gbest_fitness:
                self.gbest_fitness = fitnesses[best_fitness_idx]
                self.gbest_position = self.positions[best_fitness_idx, :]
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1

            self.iterations += 1

        return self.gbest_position


# Example usage:
if __name__ == "__main__":

    paper_width = 100
    paper_height = 100
    paper_size = (paper_width, paper_height)
    image_sizes = [[20,100],[20,100],[20,100],[20,100],[20,100]]
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
