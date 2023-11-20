import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Constants defining the image dimensions
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

def load_images_from_dat(file_path, year, image_dim):
    """
    Load images from a .dat file and reshape them according to the specified dimensions.

    Parameters:
    file_path (str): The directory containing the .dat files.
    year (int): The year of the data you want to load.
    image_dim (int): The dimension of the images (5, 20, or 60) as specified in the constants.

    Returns:
    numpy.ndarray: An array of images loaded from the .dat file.
    """
    # Construct the file name and path
    file_name = f"20d_month_has_vb_[{image_dim}]_ma_{year}_images.dat"
    full_path = os.path.join(file_path, file_name)

    # Read the .dat file as a memory-mapped array
    images = np.memmap(full_path, dtype=np.uint8, mode='r').reshape(
        (-1, IMAGE_HEIGHT[image_dim], IMAGE_WIDTH[image_dim]))
    
    return images

def display_image(image):
    """
    Display a single image using matplotlib.

    Parameters:
    image (numpy.ndarray): An array representing a single image.
    """
    plt.axis('off')  # Hide the axes
    plt.imshow(image, cmap='gray')
    plt.show()
    pass

# # Example usage:
# year_to_load = 1993
# image_dim = 20  # Assuming you want to load images of dimension 20x20
# file_path = "./"  # The path to your .dat files

# # Load images for the specified year
# images = load_images_from_dat(file_path, year_to_load, image_dim)

# # Display the first image
# if images.size > 0:
#     display_image(images[1])
# else:
#     print("No images were loaded. Please check the file path and dimensions.")


# labels_df = pd.read_feather(os.path.join("", f"20d_month_has_vb_[{image_dim}]_ma_{year_to_load}_labels_w_delay.feather"))