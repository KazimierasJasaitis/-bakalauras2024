import numpy as np
from PIL import Image, ImageDraw
import sys 

class HarmonySearch:
    def __init__(self, paper_size, image_sizes,
                 HM_size=50, desired_fitness=0, 
                 iterations_without_improvement_limit=10000, 
                 hmcr=0.8, par=0.4, 
                 pb=0.1):
        self.paper_width, self.paper_height = paper_size
        self.image_sizes = np.array(image_sizes)
        self.N = len(image_sizes)

        self.HM_size = HM_size
        self.desired_fitness = desired_fitness
        self.iterations_without_improvement_limit = iterations_without_improvement_limit        
        self.hmcr = hmcr
        self.par = par
        self.pb = pb

        self.best_position = None
        self.best_fitness = float('inf')

        self.iterations = 0
        self.iterations_without_improvement = 0

        self.lower_bound = np.zeros(self.N * 3)
        self.upper_bound = np.tile(max(paper_size), self.N * 3)

        self.HM = np.empty((HM_size, self.N, 3))
        self.HM[:, :, 0] = np.random.uniform(0, paper_size[0], (HM_size, self.N))
        self.HM[:, :, 1] = np.random.uniform(0, paper_size[1], (HM_size, self.N))
        self.HM[:, :, 2] = np.random.uniform(0.1, 1.0, (HM_size, self.N))

        self.fitnesses = np.full(HM_size, float('inf'))

    def compute_fitness_vectorized_np(self, HM, scaling_penalty_factor=20, boundary_penalty_factor=10, 
                                      overlap_penalty_factor=10, uncovered_area_penalty_factor=1):
        
        out_of_bound_penalties = np.zeros(len(HM))
        overlapping_area_penalties = np.zeros(len(HM))
        uncovered_area_penalties = np.full(len(HM), self.paper_width * self.paper_height, dtype=np.float64)

        def get_image_data(image_idx):
            X = HM[:, image_idx:image_idx+1, 0:1]
            Y = HM[:, image_idx:image_idx+1, 1:2]
            S = HM[:, image_idx:image_idx+1, 2:3]
            
            W, H = self.image_sizes[image_idx:image_idx+1, 0:1]*S, self.image_sizes[image_idx:image_idx+1, 1:2]*S 

            X1 = X
            X2 = X + W
            Y1 = Y
            Y2 = Y + H
            W, H, X1, X2, Y1, Y2 = W.round(), H.round(), X1.round(), X2.round(), Y1.round(), Y2.round()
            return W.flatten(), H.flatten(), X1.flatten(), X2.flatten(), Y1.flatten(), Y2.flatten()
        
        def calculate_scaling_penalties():
            # Calculate the average scale for each particle
            average_scales = np.mean(HM[:, :, 2], axis=1, keepdims=True)
            
            # Calculate the squared differences from the average and sum these differences for each particle
            scale_differences = HM[:, :, 2] - average_scales
            squared_differences = np.square(scale_differences)
            scaling_penalties = np.sum(squared_differences, axis=1)

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
            _, _, XA1, XA2, YA1, YA2 = get_image_data(image_idx_A)
            _, _, XB1, XB2, YB1, YB2 = get_image_data(image_idx_B)

            # Calculate horizontal and vertical overlaps
            horizontal_overlap = np.maximum(0, np.minimum(XA2, XB2) - np.maximum(XA1, XB1))
            vertical_overlap = np.maximum(0, np.minimum(YA2, YB2) - np.maximum(YA1, YB1))

            # The area of overlap is simply the product of the horizontal and vertical overlaps
            overlap_area = horizontal_overlap * vertical_overlap

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
        fitness = scaling_penalties * scaling_penalty_factor + out_of_bound_penalties * boundary_penalty_factor +\
            uncovered_area_penalties * uncovered_area_penalty_factor + overlapping_area_penalties * overlap_penalty_factor
        return fitness

    def improvise_new_harmony(self):
        new_position = np.empty((1, self.N, 3))
        for i in range(self.N * 3):
            index = i // 3
            if np.random.rand() < self.hmcr:
                new_position[0,index, i % 3] = np.random.choice(self.HM[:, index, i % 3])
                if np.random.rand() < self.par:
                    new_position[0,index, i % 3] += np.random.uniform(-1, 1) * self.pb
            else:
                new_position[0,index, i % 3] = np.random.uniform(self.lower_bound[i], self.upper_bound[i])
        new_fitness = self.compute_fitness_vectorized_np(new_position)
        return new_position, new_fitness

    def update_harmony_memory(self, new_position, new_fitness):
        worst_index = np.argmax(self.fitnesses)
        if new_fitness < self.fitnesses[worst_index]:
            self.HM[worst_index] = new_position
            self.fitnesses[worst_index] = new_fitness[0]

    def run(self):
        # Evaluate initial harmonies
        self.fitnesses = self.compute_fitness_vectorized_np(self.HM)

        while self.best_fitness > self.desired_fitness and self.iterations_without_improvement < self.iterations_without_improvement_limit:
            self.iterations_without_improvement += 1
            self.iterations += 1

            new_position, new_fitness = self.improvise_new_harmony()
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_position = new_position
                self.iterations_without_improvement = 0
                self.update_harmony_memory(new_position, new_fitness)
                sys.stdout.write('\033[K')
                print(f"New best fitness: {new_fitness[0]}", end='\r')
        return self.best_position

# Example usage
if __name__ == "__main__":
    # Setup parameters
    paper_width = 100
    paper_height = 40
    paper_size = (paper_width, paper_height)
    image_sizes = [[100,20],[100,20]]
    N = len(image_sizes)
    HM_size = 50
    desired_fitness = 0
    iterations_without_improvement_limit = 100000
    hmcr = 0.95
    par = 0.4
    pb = 0.1

    hso = HarmonySearch(paper_size=paper_size, 
                        image_sizes=image_sizes, 
                        HM_size=HM_size,
                        desired_fitness=desired_fitness, 
                        iterations_without_improvement_limit=iterations_without_improvement_limit, 
                        hmcr=hmcr, par=par, pb=pb)
    
    best_position = hso.run()

    # Print each image's position and scale factor
    best_position_2d = best_position.reshape(-1, 3)
    for i, (x, y, scale) in enumerate(best_position_2d):
        print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {round(scale, 2)}")

    print(hso.iterations)
    print(hso.best_fitness[0])
    print(hso.best_position[0])


    # Visualization (same as in PSO)
    img = Image.new('RGB', paper_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for (x, y, scale), size in zip(best_position_2d, image_sizes):
        rect = [x, y, x + size[0] * scale, y + size[1] * scale]
        draw.rectangle(rect, outline="blue", width=1)
    img.show()
