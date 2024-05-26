import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import os

sys.path.append(os.path.abspath('C:/Users/Lenovo/Desktop/Bakalauras/2023Kursinis'))
import imageCut
from imageCut import generate_image_sizes_with_solutions

def compute_fitness(y_true, y_pred, image_sizes, paper_size, min_scale, max_scale, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=5, uncovered_area_penalty_factor=5):        
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

    y_pred = y_pred.reshape(-1, 3)
    y_true = y_true.reshape(-1, 3)
    avg_scale = np.mean([scale for _, _, scale in y_pred])

    for i, (x, y, scale) in enumerate(y_pred):
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
        for j in range(i + 1, len(y_pred)):
            x2, y2, scale2 = y_pred[j]
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

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class MySimpleNN:
    def __init__(self, input_size, hidden_layer_size, output_size):
        # Initialize weights
        self.weights1 = np.random.rand(input_size, hidden_layer_size)
        self.weights2 = np.random.rand(hidden_layer_size, output_size)
        self.output_size = output_size
        
    def forward(self, X):
        self.hidden = relu(np.dot(X, self.weights1))
        output = sigmoid(np.dot(self.hidden, self.weights2))
        return output

def custom_loss_wrapper(image_sizes, paper_size, min_scale, max_scale):
    def compute_fitness_wrapper(y_true_single, y_pred_single):
        # Wrap your compute_fitness function. This assumes compute_fitness returns a scalar loss per example.
        loss = tf.py_function(compute_fitness, 
                              [y_true_single, y_pred_single, image_sizes, paper_size, min_scale, max_scale], 
                              Tout=tf.float32)
        return loss
    
    def custom_loss(y_true, y_pred):
        # Using tf.map_fn to iterate over elements in the batch
        losses = tf.map_fn(lambda x: compute_fitness_wrapper(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
        return losses
    
    return custom_loss  

def train(model, X_train, Y_train, epochs, learning_rate):
    for epoch in range(epochs):
        # Forward pass
        output = model.forward(X_train)

        # Compute loss
        loss = model.compute_loss(Y_train, output)

        # Backpropagation
        # Calculate derivative of loss w.r.t weights2 (output layer)
        d_loss_output = -(Y_train - output)
        d_output_z2 = output * (1 - output)  # Derivative of sigmoid
        d_z2_weights2 = model.a1

        # Gradient for weights2
        grad_weights2 = np.dot(d_z2_weights2.T, d_loss_output * d_output_z2)

        # Calculate derivative of loss w.r.t weights1 (hidden layer)
        d_z2_a1 = model.weights2
        d_a1_z1 = relu_derivative(model.z1)
        d_z1_weights1 = X_train

        # Gradient for weights1
        grad_weights1 = np.dot(d_z1_weights1.T, np.dot(d_loss_output * d_output_z2, d_z2_a1.T) * d_a1_z1)

        # Update weights
        model.weights1 -= learning_rate * grad_weights1
        model.weights2 -= learning_rate * grad_weights2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

  

# Example usage:
if __name__ == "__main__":
    N = 5  # Number of images/solutions per sample
    paper_size = (100, 100)
    m = 40000
    # Check if the model file exists
    if os.path.exists(f"./models/nn_N{N}_m{m}.h5"):
        # Load the model from the file
        model = load_model(f"./models/nn_N{N}_m{m}.h5")
        print("Model loaded successfully from file.")
    else:
        # Initialize lists to hold all the samples
        X_train = []
        Y_train = []

        for _ in range(m):
            image_sizes, solutions_dict = imageCut.generate_image_sizes_with_solutions(N, paper_size)
            
            # Flatten the image_sizes and solutions for this sample
            # For image_sizes: Flatten each pair of dimensions into a single array
            x_sample = np.array(image_sizes).flatten()
            
            # For solutions: Convert dictionary to array and flatten x, y, scale into a single array
            y_sample = np.array([(sol['x'], sol['y'], sol['scale']) for sol in solutions_dict]).flatten()
            
            # Append to your training data
            X_train.append(x_sample)
            Y_train.append(y_sample)

        # Convert lists to numpy arrays for neural network training
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Calculate total x and y lengths of all images
        sum_x_lengths = np.sum([size[0] for size in image_sizes])
        sum_y_lengths = np.sum([size[1] for size in image_sizes])
        paper_height, paper_width = paper_size
        max_scale = max(paper_height,paper_width)/min(val for sublist in image_sizes for val in sublist)
        min_scale = min(paper_height/sum_y_lengths,paper_width/sum_x_lengths)

        # Create the model
        model = create_model(X_train, Y_train, image_sizes, paper_size, min_scale, max_scale)

        ################################MODEL TRAINING################################
        model.fit(X_train, Y_train, epochs=10, batch_size=10)

        model.save(f"./models/nn_N{N}_m{m}.h5")  # Save the model to a file
    ###########################MODEL TEST################################
    # Example input_vector
    input_vector = [(100, 42), (18, 58), (12, 58), (70, 17), (70, 41)]

    # Flatten the input_vector and convert to a NumPy array
    input_array = np.array([element for tuple in input_vector for element in tuple])

    # Reshape the array if necessary to match the input shape expected by the model
    # This step depends on the specific input structure your model expects
    # For example, if your model expects a single sample with a specific shape, you might need:
    input_array = input_array.reshape(1, -1)  # Reshape to 1 sample, with the appropriate number of features

    # Now you can use the reshaped array in model.predict
    prediction = model.predict(input_array)

    predicted_positions_and_scales = prediction.reshape((N, 3))  # Reshape to (num_images, 3)

    # Now, predicted_positions_and_scales contains a list where each item is [x, y, scale] for an image
    for i, (x, y, scale) in enumerate(predicted_positions_and_scales):
        print(f"Image {i+1}: x = {x}, y = {y}, scale = {scale}")


    from PIL import Image, ImageDraw

    # Create a blank canvas
    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw each rectangle
    for (x, y, scale), size in zip(predicted_positions_and_scales, input_vector):
        rect = [x, y, x + size[0] * scale, y + size[1] * scale]
        draw.rectangle(rect, outline ="blue", width=1)

    # Show the image
    img.show()
