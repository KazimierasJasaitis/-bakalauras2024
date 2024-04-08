import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    N = 5
    paper_size = (100, 100)
    image_sizes = generate_image_sizes(N, paper_size)
    print(image_sizes)
    display_images(image_sizes, paper_size)

