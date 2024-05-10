import cv2
import os
import random

# Define the paths for the input and output folders
input_folder = "/home/pavaniankam/Downloads/cartzy_sample_data/89009710/"
output_folder_1 = "/home/pavaniankam/Downloads/cartzy_sample_data/80824107/"
output_folder_2 = "/home/pavaniankam/Downloads/val10/"
output_folder_3 = "/home/pavaniankam/Downloads/testing10/"

# Get the list of image files in the input folder
image_files = os.listdir(input_folder)

# Calculate the number of images for each category
total_images = len(image_files)
category_1_count = int(total_images * 0.8)
category_2_count = int(total_images * 0.1)
category_3_count = total_images - category_1_count - category_2_count

# Shuffle the image files randomly
random.shuffle(image_files)

# Move the images to the respective folders based on the distribution
for i, image_file in enumerate(image_files):
    image_path = os.path.join(input_folder, image_file)
    if i < category_1_count:
        output_path = os.path.join(output_folder_1, image_file)
    elif i < category_1_count + category_2_count:
        output_path = os.path.join(output_folder_2, image_file)
    else:
        output_path = os.path.join(output_folder_3, image_file)
    os.rename(image_path, output_path)

print("Images moved successfully!")