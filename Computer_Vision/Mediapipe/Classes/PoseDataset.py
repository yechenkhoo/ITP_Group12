import seaborn as sns
import keras
import pandas as pd
from keras import layers, Sequential
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os


class PoseDataset():
    def __init__(self, path):
        self.path = path
        
    def load_csv_data(self):
        """
        Load CSV and extract attributes required.
        """
        self.df = pd.read_csv(self.path)
        self.allClasses = sorted(self.df['Pose_Class'].unique())
        self.classCount = len(self.allClasses)
    
    def split_dataset(self, test_size=0.2, reshape=False, random_state=0):
        """
        Splits the loaded dataset into training, validation, and test set.

        Parameters:
        - test_size: determines the split ratio of dataset between training and test.
        The validation set will be test_size% of the non-test set.
        e.g. if test_size is 0.2, training set is 0.64 and validation set is 0.16.

        - reshape: reshapes the data into 33x4 instead of 132x1
        """
        x = self.df.copy()

        # Sort by Pose_Class from P1 to P10
        x['Pose_Class'] = pd.Categorical(x['Pose_Class'], categories=[f'P{i}' for i in range(1, 11)], ordered=True)
        x = x.sort_values('Pose_Class')

        image_paths = x.pop('Image_Path').to_list()  # store image paths
        # x = x.drop(columns=[col for col in x.columns if '_Z' in col or '_V' in col]) # to test x and y only
        y, _ = x.pop('Pose_Class').factorize()
        print(_)
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Converts it back to 33x4 instead of 132x1
        if reshape:
            x_np = x.to_numpy()
            print(f"Before reshape: {x_np.shape}")
            x = self.reshape_keypoints(x_np)
        print(f"After reshape: {x.shape}")

        # Split full dataset into train+validation set and test set
        x_trainval, self.x_test, y_trainval, self.y_test, paths_trainval, self.paths_test = train_test_split(
            x, y, image_paths, test_size=test_size, random_state=random_state)

        # Split train+validation into train set and val set
        self.x_train, self.x_val, self.y_train, self.y_val, self.paths_train, self.paths_val = train_test_split(
            x_trainval, y_trainval, paths_trainval, test_size=test_size, random_state=random_state)
        
        print(f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test: {len(self.x_test)}")

    def split_dataset_manual(self, val_size=0.2, reshape=False):
        """
        Manually splits the dataset:
        - Test set: rows where 'Image_Path' contains 'Charlie'
        - Train/Val: all other rows
        - Validation set will be val_size fraction of train set

        Parameters:
        - val_size: determines how much of the non-test set is used for validation
        - reshape: reshapes the data into 33x4 instead of 132x1
        """
        x = self.df.copy()

        # Sort by Pose_Class from P1 to P10
        x['Pose_Class'] = pd.Categorical(x['Pose_Class'], categories=[f'P{i}' for i in range(1, 11)], ordered=True)
        x = x.sort_values('Pose_Class')

        image_paths = x.pop('Image_Path').to_list()

        # Extract labels
        y, _ = x.pop('Pose_Class').factorize()
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Convert to numpy for easier indexing
        x_np = x.to_numpy()
        y_np = y
        paths_np = np.array(image_paths)

        # Get indices for Charlie images
        test_indices = [i for i, p in enumerate(paths_np) if "Charlie" in p]
        trainval_indices = [i for i in range(len(paths_np)) if i not in test_indices]

        # Subset arrays
        self.x_test = x_np[test_indices]
        self.y_test = y_np[test_indices]
        self.paths_test = paths_np[test_indices]

        x_trainval = x_np[trainval_indices]
        y_trainval = y_np[trainval_indices]
        paths_trainval = paths_np[trainval_indices]

        # Reshape if needed
        if reshape:
            print(f"Before reshape: {x_trainval.shape}, {self.x_test.shape}")
            x_trainval = self.reshape_keypoints(x_trainval)
            self.x_test = self.reshape_keypoints(self.x_test)
        print(f"After reshape: {x_trainval.shape}, {self.x_test.shape}")

        # Split trainval into train and val
        self.x_train, self.x_val, self.y_train, self.y_val, self.paths_train, self.paths_val = train_test_split(
            x_trainval, y_trainval, paths_trainval, test_size=val_size, random_state=0)

        print(f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test (Charlie): {len(self.x_test)}")


    
    def reshape_keypoints(self, X_flat):
        """
        # Converts a flat input like (132,) to a structured (33, 4) shape.
        # Assumes 33 keypoints each with 4 features: x, y, z, visibility.
        # Works for a single sample or batch of samples.
        """
        if X_flat.ndim == 1:
            # return X_flat.reshape((33, 2))
            return X_flat.reshape((33, 4))
        elif X_flat.ndim == 2:
            # return X_flat.reshape((-1, 33, 2))
            return X_flat.reshape((-1, 33, 4))
        else:
            raise ValueError("Input must be a 1D or 2D array")