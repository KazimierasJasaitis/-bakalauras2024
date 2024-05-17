import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.initializers import HeNormal, random_uniform
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append('./2023Kursinis')  # Adjust the path as necessary
import imageCut # type: ignore
from imageCut import generate_image_sizes_with_solutions_6N_input_features # type: ignore


# def create_model(X_train, Y_train):
#     model = Sequential([
#         Dense(512, activation='relu', kernel_initializer=random_uniform(), input_shape=(len(X_train[0]),)),
#         Dense(256, activation='relu'),
#         Dense(512, activation='relu'),
#         Dense(len(Y_train[0]), activation='linear')
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

#     return model

def create_model(X_train, Y_train):
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    model = Sequential([
        Dense(512, activation='relu', kernel_initializer=random_uniform(), input_shape=(X_train_scaled.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(Y_train.shape[1], activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.000001), loss="mse", metrics=["mae"])

    return model, scaler

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

if __name__ == "__main__":
    N = 3
    image_feature_N = 8
    constant_features = [N]
    other_feature_N = 12
    paper_size = (100, 100)
    m = 10000
    version = 8

    new = False
    if len(sys.argv) > 1:
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
        model, scaler = create_model(X_train, Y_train)

        ################################ MODEL TRAINING ################################
        model.fit(X_train, Y_train, epochs=100, batch_size=1)

        model.save(f"./models/nn_N{N}_m{m}_{version}.keras")

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