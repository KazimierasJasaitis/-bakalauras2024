import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np

def split_image(image_size, horizontal=True):
    if horizontal:
        split_point = random.randint(1, image_size[1] - 1)
        return [(image_size[0], split_point), (image_size[0], image_size[1] - split_point)]
    else:
        split_point = random.randint(1, image_size[0] - 1)
        return [(split_point, image_size[1]), (image_size[0] - split_point, image_size[1])]

def generate_image_sizes(N, paper_size):
    if N <= 0 or not all(isinstance(d, int) and d > 0 for d in paper_size):
        raise ValueError("Invalid input: N must be positive and paper_size must be a tuple of positive integers.")

    image_sizes = [paper_size]

    while len(image_sizes) < N:
        # Randomly select an image to split
        selected_index = random.randint(0, len(image_sizes) - 1)
        selected_image = image_sizes.pop(selected_index)

        # Randomly choose between a horizontal or vertical split
        new_images = split_image(selected_image, horizontal=random.choice([True, False]))
        image_sizes.extend(new_images)

    return image_sizes

def generate_image_sizes_with_solutions(N, paper_size):
    if N <= 0 or not all(isinstance(d, int) and d > 0 for d in paper_size):
        raise ValueError("Invalid input: N must be positive and paper_size must be a tuple of positive integers.")
    
    # Initialize the list with the initial paper size and position (0,0)
    images = [{'size': paper_size, 'pos': (0, 0), 'scale': 100}]
    
    while len(images) < N:
        selected_index = random.randint(0, len(images) - 1)
        selected_image = images.pop(selected_index)
        
        # Determine new images after split, including their positions
        new_images = split_image_with_pos(selected_image, horizontal=random.choice([True, False]))
        images.extend(new_images)
    
    # Extract the required output format
    image_sizes = [img['size'] for img in images]
    solutions = [(img['pos'][0], img['pos'][1], img['scale']) for img in images]

    return image_sizes, solutions


def generate_image_sizes_with_solutions_6N_input_features(N, paper_size):
    if N <= 0 or not all(isinstance(d, int) and d > 0 for d in paper_size):
        raise ValueError("Invalid input: N must be positive and paper_size must be a tuple of positive integers.")
    
    images = [{'size': paper_size, 'pos': (0, 0), 'scale': 100}]
    
    while len(images) < N:
        selected_index = random.randint(0, len(images) - 1)
        selected_image = images.pop(selected_index)
        
        # Determine new images after split, including their positions
        new_images = split_image_with_pos(selected_image, horizontal=random.choice([True, False]))
        images.extend(new_images)
    
    # Extract the required output format
    image_sizes = [
        (img['size'][0], 
         img['size'][1],
         paper_size[0] - img['size'][0],
         paper_size[1] - img['size'][1],
        (paper_size[0] / img['size'][0]), 
        (paper_size[1] / img['size'][1]),
        (img['size'][0] * img['size'][1]),
        (img['size'][0] / img['size'][1]))
        for img in images
    ]

    solutions = [(img['pos'][0], img['pos'][1], img['scale']) for img in images]

    return image_sizes, solutions


def split_image_with_pos(image_info, horizontal=True):
    image_size, (x, y) = image_info['size'], image_info['pos']

    # Adjust split direction if one of the dimensions is 1
    if image_size[0] == 1:  # Only height can be split
        horizontal = False
    elif image_size[1] == 1:  # Only width can be split
        horizontal = True

    if horizontal:
        # Prevent splitting if dimension is already at minimum
        if image_size[1] <= 1:
            return [image_info]  # Return the image as is, no split possible
        split_point = random.randint(1, image_size[1] - 1)
        first_half = {'size': (image_size[0], split_point), 'pos': (x, y), 'scale': 100}
        second_half = {'size': (image_size[0], image_size[1] - split_point), 'pos': (x, y + split_point), 'scale': 100}
    else:
        # Prevent splitting if dimension is already at minimum
        if image_size[0] <= 1:
            return [image_info]  # Return the image as is, no split possible
        split_point = random.randint(1, image_size[0] - 1)
        first_half = {'size': (split_point, image_size[1]), 'pos': (x, y), 'scale': 100}
        second_half = {'size': (image_size[0] - split_point, image_size[1]), 'pos': (x + split_point, y), 'scale': 100}

    return [first_half, second_half]



def display_images(image_sizes, paper_size):
    fig, axs = plt.subplots(1, len(image_sizes), figsize=(10, 5))

    for ax, size in zip(axs, image_sizes):
        ax.add_patch(patches.Rectangle((0, 0), size[0], size[1], edgecolor='black', facecolor='none'))
        ax.text(size[0]/2, size[1]/2, f'{size[0]}x{size[1]}', horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, max(width for width, _ in image_sizes))
        ax.set_ylim(0, max(height for _, height in image_sizes))
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    N = 3
    paper_size = (100, 100)
    image_sizes, solutions = generate_image_sizes_with_solutions(N, paper_size)
    print(image_sizes,solutions)
    display_images(image_sizes, paper_size)


    # Assuming generate_image_sizes_with_solutions is correctly implemented and called here
    image_sizes, solutions = generate_image_sizes_with_solutions(N, paper_size)

    # Since solutions is assumed to be already in the right format, no need to convert it
    solutions_np = np.array(solutions)

    # Create a blank canvas
    img = Image.new('RGB', paper_size, color = (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for (x, y, scale), size in zip(solutions_np, image_sizes):
        # Calculate the rectangle coordinates based on x, y, scale, and original image size
        rect = [x, y, x + size[0] * scale/100, y + size[1] * scale/100]
        print(rect)
        # Draw the rectangle on the image
        draw.rectangle(rect, outline="blue", width=1)

    # Display the image
    img.show()
