import os
import shutil
import random
import argparse

'''
This script helps to split datasets into train test splits 

Example usage: python split.py fo_dataset

follow this structure:
        fo_dataset
        |__ P1
            |______ 00000128.jpg
            |______ 00000181.jpg
            |______ ...
        |__ P2
            |______ 00000243.jpg
            |______ 00000306.jpg
            |______ ...
        ...

Args:
    split_dir: Path to the directory of the images to split into train test splits
'''


def main(image_dir):
    # Define the percentage of images for training data
    train_percentage = 0.7

    # Create the train and test directories if they don't exist
    train_dir = os.path.join(image_dir, "train")
    test_dir = os.path.join(image_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each folder in the image directory
    for folder in os.listdir(image_dir):
        # Get the full path of the folder
        folder_path = os.path.join(image_dir, folder)

        # Check if it's a directory and not the train or test directory
        if os.path.isdir(folder_path) and folder not in ["train", "test"]:
            # Create the corresponding train and test subfolders
            train_subfolder = os.path.join(train_dir, folder)
            test_subfolder = os.path.join(test_dir, folder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)
            # Get a list of images in the folder
            images = os.listdir(folder_path)

            # Calculate the number of images for training data
            num_train_images = int(len(images) * train_percentage)

            # Move the training images to the train directory
            for i in range(num_train_images):
                image_path = os.path.join(folder_path, images[i])
                shutil.move(image_path, os.path.join(train_dir, folder, images[i]))

            # Move the remaining images to the test directory
            for i in range(num_train_images, len(images)):
                image_path = os.path.join(folder_path, images[i])
                shutil.move(image_path, os.path.join(test_dir, folder, images[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data splitting for images.')
    parser.add_argument('split_dir', type=str, help='Path to the base directory of original images for train and test split.')

    args = parser.parse_args()
    main(args.split_dir)                