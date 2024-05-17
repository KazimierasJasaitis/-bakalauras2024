import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.initializers import HeNormal
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append('./2023Kursinis')  # Adjust the path as necessary
import imageCut # type: ignore
from imageCut import generate_image_sizes_with_solutions # type: ignore

def compute_fitness(y_pred, image_sizes, paper_size, min_scale, max_scale, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=5, uncovered_area_penalty_factor=5):        
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


    positions = y_pred.numpy()
    # positions = y_pred
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

def compute_fitness_vectorized_np(y_pred, X_train, paper_size, image_feature_N,scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=100, uncovered_area_penalty_factor=1):
    out_of_bound_penalty = 0
    overlapping_area_penalty = 0
    uncovered_area_penalty = paper_size[0] * paper_size[1]

    N = len(y_pred[0]) // 3
    # print(y_pred)
    # print(X_train)

    def calculate_out_of_bound_penalty_for_image(y_pred, X_train, paper_size, image_index):
        # Calculate column indices for the given image
        start_idx = image_index * 3
        X = y_pred[:, start_idx:start_idx+1]
        Y = y_pred[:, start_idx+1:start_idx+2]
        S = y_pred[:, start_idx+2:start_idx+3]/100
        
        # Assuming X_train has dimensions [batch_size, N, 2] and contains the original width and height for each image
        W, H = X_train.reshape(-1, N, 2)[:, image_index,0]*S, X_train.reshape(-1, N, 2)[:,image_index,1]*S
        
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
    
    def calculate_area_for_image(y_pred, X_train, paper_size, image_index):
        # Calculate column indices for the given image
        start_idx = image_index * 3
        X = y_pred[:, start_idx:start_idx+1]
        Y = y_pred[:, start_idx+1:start_idx+2]
        S = y_pred[:, start_idx+2:start_idx+3]/100
        
        # Assuming X_train has dimensions [batch_size, N, 2] and contains the original width and height for each image
        W, H = X_train.reshape(-1, N, 2)[:, image_index,0]*S, X_train.reshape(-1, N, 2)[:,image_index,1]*S
        
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
    
    def calculate_overlap(y_pred, X_train, image_index_A, image_index_B):
        start_idx_A = image_index_A * 3
        start_idx_B = image_index_B * 3
        
        XA = y_pred[:, start_idx_A:start_idx_A+1]
        XB = y_pred[:, start_idx_B:start_idx_B+1]

        YA = y_pred[:, start_idx_A+1:start_idx_A+2]
        YB = y_pred[:, start_idx_B+1:start_idx_B+2]

        SA = y_pred[:, start_idx_A+2:start_idx_A+3]/100
        SB = y_pred[:, start_idx_B+2:start_idx_B+3]/100

        # Assuming X_train has dimensions [batch_size, N, 2]
        WA, HA = X_train.reshape(-1, N, 2)[:, image_index_A,0]*SA, X_train.reshape(-1, N, 2)[:,image_index_A,1]*SA
        WB, HB = X_train.reshape(-1, N, 2)[:, image_index_B,0]*SB, X_train.reshape(-1, N, 2)[:,image_index_B,1]*SB
        
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

        overlap = tf.maximum(tf.where(XA1 < XB1, 0.0, tf.maximum(WB - (XA1 - XB1), 0.0)) + 
                            tf.where(XB2 < XA2, 0.0, tf.maximum(WB - (XB2 - XA2), 0.0)), 0.0) * \
                tf.maximum(tf.where(YA1 < YB1, 0.0, tf.maximum(HB - (YA1 - YB1), 0.0)) + 
                            tf.where(YB2 < YA2, 0.0, tf.maximum(HB - (YB2 - YA2), 0.0)), 0.0)
        if (overlap > 0):             
            print(f"Overlap: {overlap}, image_index_A: {image_index_A}, image_index_B: {image_index_B}")
            print(f"XA: {XA}, YA: {YA}, SA: {SA}")
            print(f"WA: {WA}, H: {HA}")
            print(f"XB: {XB}, YB: {YB}, SB: {SB}")
            print(f"WB: {WB}, HB: {HB}")
            print(" ")
        overlap_penalty = np.maximum(overlap, 0.0)

        return overlap_penalty

    for i in range(N):
        out_of_bound_penalty += calculate_out_of_bound_penalty_for_image(y_pred, X_train, paper_size, i)
        for j in range(i+1, N):
            overlapping_area_penalty += calculate_overlap(y_pred, X_train, i, j)
            # print(overlapping_area_penalty)
            # print(i,j)
            # print(" ")
        # Reiktų pridėt overlapping area penalty tik tom vietom kurios yra inbound
        uncovered_area_penalty -= calculate_area_for_image(y_pred, X_train, paper_size, i) 
        
    uncovered_area_penalty = np.maximum(uncovered_area_penalty, 0.0)

    print(f"Out of bound penalty: {out_of_bound_penalty}")
    print(f"Overlapping area penalty: {overlapping_area_penalty}")
    print(f"Uncovered area penalty: {uncovered_area_penalty}")
    print(" ")
    fitness = out_of_bound_penalty * boundary_penalty_factor + uncovered_area_penalty * uncovered_area_penalty_factor +\
              overlapping_area_penalty * overlap_penalty_factor
    return np.mean(fitness)


def compute_fitness_vectorized(y_pred, X_train, paper_size, image_feature_N, scaling_penalty_factor=1, boundary_penalty_factor=10, overlap_penalty_factor=100, uncovered_area_penalty_factor=1):
    out_of_bound_penalty = 0
    overlapping_area_penalty = 0
    uncovered_area_penalty = paper_size[0] * paper_size[1]
    N = 3
    unused_feature_N = 12
    X_train = X_train[:, :-(unused_feature_N)]
    X_train = tf.cast(X_train, tf.float32)
    X_train = tf.reshape(X_train, [-1, N, image_feature_N])

    def calculate_out_of_bound_penalty_for_image(y_pred, X_train, paper_size, image_index):
        # Calculate column indices for the given image
        start_idx = image_index * 3
        X = y_pred[:, start_idx:start_idx+1]
        Y = y_pred[:, start_idx+1:start_idx+2]
        S = y_pred[:, start_idx+2:start_idx+3]/100
        
        # Assuming X_train has dimensions [batch_size, N, 2] and contains the original width and height for each image
        W, H = tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index, 0] * S, tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index, 1] * S
        
        paper_h, paper_w = paper_size
        
        X1 = X
        X2 = X + W
        Y1 = Y
        Y2 = Y + H
        
        area = tf.maximum(W * H, 0.0)
        inbound = tf.maximum(paper_w - tf.maximum(X1, 0.0) - tf.maximum(paper_w - tf.maximum(X2, 0.0), 0.0), 0.0) * tf.maximum(paper_h - tf.maximum(Y1, 0.0) - tf.maximum(paper_h - tf.maximum(Y2, 0.0), 0.0), 0.0)
        
        out_of_bound_penalty = tf.maximum(area - inbound, 0.0)

        negative_penalty = tf.abs(tf.minimum(X, 0.0)) + tf.abs(tf.minimum(Y, 0.0))+ tf.abs(tf.minimum(S, 0.0))*1000
        
        return out_of_bound_penalty + negative_penalty
    
    def calculate_area_for_image(y_pred, X_train, paper_size, image_index):
        # Calculate column indices for the given image
        start_idx = image_index * 3
        X = y_pred[:, start_idx:start_idx+1]
        Y = y_pred[:, start_idx+1:start_idx+2]
        S = y_pred[:, start_idx+2:start_idx+3]/100
        
        # Assuming X_train has dimensions [batch_size, N, 2] and contains the original width and height for each image
        W, H = tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index, 0] * S, tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index, 1] * S
        
        paper_h, paper_w = paper_size
        
        X1 = X
        X2 = X + W
        Y1 = Y
        Y2 = Y + H
        
        inbound = tf.maximum(paper_w - tf.maximum(X1, 0.0) - tf.maximum(paper_w - tf.maximum(X2, 0.0), 0.0), 0.0) * tf.maximum(paper_h - tf.maximum(Y1, 0.0) - tf.maximum(paper_h - tf.maximum(Y2, 0.0), 0.0), 0.0)

        return inbound

    def calculate_overlap(y_pred, X_train, image_index_A, image_index_B):
        start_idx_A = image_index_A * 3
        start_idx_B = image_index_B * 3
        
        XA = y_pred[:, start_idx_A:start_idx_A+1]
        XB = y_pred[:, start_idx_B:start_idx_B+1]

        YA = y_pred[:, start_idx_A+1:start_idx_A+2]
        YB = y_pred[:, start_idx_B+1:start_idx_B+2]

        SA = y_pred[:, start_idx_A+2:start_idx_A+3]/100
        SB = y_pred[:, start_idx_B+2:start_idx_B+3]/100

        # Assuming X_train has dimensions [batch_size, N, 2]
        WA, HA = tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index_A, 0] * SA, tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index_A, 1] * SA
        WB, HB = tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index_B, 0] * SB, tf.reshape(X_train, [-1, N, image_feature_N])[:, image_index_B, 1] * SB

        XA1 = XA
        XB1 = XB

        XA2 = XA + WA
        XB2 = XB + WB

        YA1 = YA
        YB1 = YB

        YA2 = YA + HA
        YB2 = YB + HB
        

        overlap = tf.maximum(tf.maximum(XA1-XB1, 0.0) - tf.maximum(WB - tf.maximum(XB2-XA2, 0.0), 0.0), 0.0) * \
                  tf.maximum(tf.maximum(YB1-YA1, 0.0) - tf.maximum(HB - tf.maximum(YA2-YB2, 0.0), 0.0), 0.0)

        # Calculate the out-of-bound penalty
        overlap_penalty = tf.maximum(overlap, 0.0)

        return overlap_penalty

    for i in range(N):
        out_of_bound_penalty += calculate_out_of_bound_penalty_for_image(y_pred, X_train, paper_size, i)
        for j in range(i+1, N):
            overlapping_area_penalty += calculate_overlap(y_pred, X_train, i, j)
        # Reiktų pridėt uncovered area penalty tik tom vietom kurios yra inbound
        uncovered_area_penalty -= calculate_area_for_image(y_pred, X_train, paper_size, i) 

    uncovered_area_penalty = tf.maximum(uncovered_area_penalty, 0.0)

    fitness = out_of_bound_penalty * boundary_penalty_factor + uncovered_area_penalty * uncovered_area_penalty_factor +\
              overlapping_area_penalty * overlap_penalty_factor

    return tf.reduce_mean(fitness)


def custom_loss(X_train_batch, y_true, y_pred, paper_size, image_feature_N):
    main_penalty = compute_fitness_vectorized(y_pred, X_train_batch, paper_size, image_feature_N)
    return main_penalty


def create_model(X_train, Y_train, paper_size, image_feature_N):
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)
    model = CustomModel(input_shape=(X_train_scaled.shape[1],), output_units=Y_train.shape[1], paper_size=paper_size, image_feature_N=image_feature_N)
    model.compile(optimizer=Adam(learning_rate=0.001))
    return model, scaler

@tf.keras.utils.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, input_shape, output_units, paper_size, image_feature_N, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.dense1 = Dense(10, activation='relu', input_shape=(input_shape,), kernel_initializer=HeNormal())
        self.dense2 = Dense(16, activation='relu')
        self.dense3 = Dense(16, activation='relu')
        self.dense4 = Dense(16, activation='relu')
        self.dense5 = Dense(output_units, activation='linear')

        self.paper_size = paper_size
        self.image_feature_N = image_feature_N
        self.input_shape = input_shape
        self.output_units = output_units

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = custom_loss(x, y, y_pred, self.paper_size, self.image_feature_N)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
    
    def get_config(self):
        # Return all initialization parameters of the model
        config = super(CustomModel, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'output_units': self.output_units,
            'paper_size': self.paper_size,
            'image_feature_N': self.image_feature_N,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Remove the extra parameters that are not required by the constructor
        if 'name' in config:
            del config['name']
        if 'trainable' in config:
            del config['trainable']
        if 'dtype' in config:
            del config['dtype']
        return cls(**config)


def get_data(N, paper_size, constant_features, m=1):
        X_train = []
        Y_train = []
        for _ in range(m):
            image_sizes, solutions = imageCut.generate_image_sizes_with_solutions_6N_input_features(N, paper_size)
            average_width = np.mean([image[0] for image in image_sizes])
            average_height = np.mean([image[1] for image in image_sizes])
            average_area = np.mean([image[0] * image[1] for image in image_sizes])

             # Calculate width and height ratios
            max_width = max(image[0] for image in image_sizes)
            min_width = min(image[0] for image in image_sizes)
            max_height = max(image[1] for image in image_sizes)
            min_height = min(image[1] for image in image_sizes)
            width_ratio = max_width / min_width if min_width > 0 else 0
            height_ratio = max_height / min_height if min_height > 0 else 0
            width_sum_1 = image_sizes[0][0] + image_sizes[1][0]
            width_sum_2 = image_sizes[0][0] + image_sizes[2][0]
            width_sum_3 = image_sizes[1][0] + image_sizes[2][0]
            height_sum_1 = image_sizes[0][1] + image_sizes[1][1]
            height_sum_2 = image_sizes[0][1] + image_sizes[2][1]
            height_sum_3 = image_sizes[1][1] + image_sizes[2][1]

            other_features = [average_width, average_height, average_area, width_ratio, height_ratio, width_sum_1, width_sum_2, width_sum_3, height_sum_1, height_sum_2, height_sum_3]

            # append constant features and other features
            
            x_sample = np.append(np.array(image_sizes).flatten(),np.array(constant_features+other_features))
            y_sample = np.array(solutions).flatten()
            X_train.append(x_sample)
            Y_train.append(y_sample)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

# Example usage:
if __name__ == "__main__":
    N = 3
    image_feature_N = 8
    constant_features = [N]
    other_feature_N = 12
    paper_size = (100, 100)
    m = 10000
    version = 9

    new = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "-n":
            new = True
        else:
            version = int(sys.argv[1])
            new = True

    if os.path.exists(f"./models/nn_N{N}_m{m}_{version}.keras") and not new:
        model = load_model(
            f"./models/nn_N{N}_m{m}_{version}.keras",
            compile=False
        )
        print("Model loaded successfully from file.")
    else:
        X_train, Y_train = get_data(N, paper_size,constant_features, m)

        model, scaler = create_model(X_train, Y_train, paper_size, image_feature_N)

        ################################MODEL TRAINING################################
        model.fit(X_train, Y_train, epochs=10, batch_size=1)

        model.save(f"./models/nn_N{N}_m{m}_{version}.keras")  # Save the model to a file

        ########################### MODEL TEST ################################
    input_array, solution = get_data(N, paper_size, constant_features, 1)
    if 'scaler' in locals():  # Check if scaler is defined
        input_array_scaled = scaler.transform(input_array)
        prediction = model.predict(input_array_scaled)
    else:
        prediction = model.predict(input_array)
    predicted_positions_and_scales = prediction.reshape((N, 3))

    for i, (x, y, scale) in enumerate(predicted_positions_and_scales):
        x = round(x)
        y = round(y)
        scale = round(scale)/100
        print(f"Image {i+1}: x = {x}, y = {y}, scale = {scale}")
    
    print("______________________________")
    for i, (x, y, scale) in enumerate(solution[0].reshape((N, 3))):
        x = round(x)
        y = round(y)
        scale = round(scale)/100
        print(f"Image {i+1}: x = {x}, y = {y}, scale = {scale}")
    
    from PIL import Image, ImageDraw
    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i, (x, y, scale) in enumerate(predicted_positions_and_scales):
        width = input_array[0][(image_feature_N) * i]
        height = input_array[0][(image_feature_N) * i + 1]
        x = round(x)
        y = round(y)
        scale = round(scale) / 100  # Adjust scale
        rect = [x, y, x + int(width * scale), y + int(height * scale)]
        draw.rectangle(rect, outline="blue", width=1)
    img.show()

    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, (x, y, scale) in enumerate(solution[0].reshape((N, 3))):
        width = input_array[0][(image_feature_N) * i]
        height = input_array[0][(image_feature_N) * i + 1]
        print(width, height, x, y, scale)
        x = round(x)
        y = round(y)
        scale = round(scale) / 100  # Adjust scale
        rect = [x, y, x + int(width * scale), y + int(height * scale)]
        draw.rectangle(rect, outline="blue", width=1)
    img.show()