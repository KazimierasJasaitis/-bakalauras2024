import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def custom_loss(y_true, y_pred, image_sizes, paper_size, min_scale, max_scale, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=5, uncovered_area_penalty_factor=5):        
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

def create_model(input_vector_size, output_vector_size):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_vector_size,)),
        Dense(64, activation='relu'),
        Dense(output_vector_size, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model



# Example usage:
if __name__ == "__main__":

    paper_width = 100
    paper_height = 150
    paper_size = (paper_width, paper_height)
    image_sizes = [[1000,500],[1000,500],[500,500],[500,500]]

    X_train = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    Y_train = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    # Flatten the image dimensions and prepend the paper size
    X_train = [paper_width, paper_height] + [dim for image in image_sizes for dim in image]

    X_train = np.array([X_train])  # Shape should be (1, N) where N is the length of the input vector
    model = create_model(input_vector_size=X_train.shape[1], output_vector_size=Y_train.shape[1])

    # Fit the model on the dataset
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

    input_vector = [[1000,500],[1000,500],[500,500],[500,500]]
    prediction = model.predict(input_vector)

    # Reshape or process the prediction as needed
    # If the output is a flattened vector: [x1, y1, scale1, x2, y2, scale2, ...]
    num_images = len(image_sizes)
    predicted_positions_and_scales = prediction.reshape((num_images, 3))  # Reshape to (num_images, 3)

    # Now, predicted_positions_and_scales contains a list where each item is [x, y, scale] for an image
    for i, (x, y, scale) in enumerate(predicted_positions_and_scales):
        print(f"Image {i+1}: x = {x}, y = {y}, scale = {scale}")


    from PIL import Image, ImageDraw

    # Create a blank canvas
    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw each rectangle
    for (x, y, scale), size in zip(predicted_positions_and_scales, image_sizes):
        rect = [x, y, x + size[0] * scale, y + size[1] * scale]
        draw.rectangle(rect, outline ="blue", width=1)

    # Show the image
    img.show()
