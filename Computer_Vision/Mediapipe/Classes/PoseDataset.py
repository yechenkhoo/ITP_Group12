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
        # Sort Pose_Class numerically by the number after 'P'
        self.allClasses = sorted(
            self.df['Pose_Class'].unique(),
            key=lambda x: int(x[1:]) if x[1:].isdigit() else x
        )

        # Set as ordered categorical
        self.df['Pose_Class'] = pd.Categorical(
            self.df['Pose_Class'],
            categories=self.allClasses,
            ordered=True
        )

        # Sort the DataFrame based on the ordered category
        self.df = self.df.sort_values('Pose_Class')

        # Refresh class info
        self.allClasses = self.df['Pose_Class'].unique()
        self.classCount = len(self.allClasses)

    def split_dataset(self, test_size=0.2, reshape=False, random_state=0):
        """
        Splits the loaded dataset into training, validation, and test set.

        Parameters:
        - test_size: determines the split ratio of dataset between training and test.
        - reshape: reshapes the data into 33x4 instead of 132x1
        """
        x = self.df.copy()

        # Sort by Pose_Class from P1 to P10
        x['Pose_Class'] = pd.Categorical(x['Pose_Class'], categories=[f'P{i}' for i in range(1, 11)], ordered=True)
        x = x.sort_values('Pose_Class')

        image_paths = x.pop('Image_Path').to_list()  # store image paths
        y, _ = x.pop('Pose_Class').factorize()
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Converts it back to 33x4 instead of 132x1
        if reshape:
            x_np = x.to_numpy()
            print(f"Before reshape: {x_np.shape}")
            x = self.reshape_keypoints(x_np)
        print(f"After reshape: {x.shape}")

        # FOR CROSS-VALIDATION STABILITY CHECK (across whole dataset)
        if test_size == 0.0:
            # All data is train, nothing in val/test
            self.x_train = x
            self.y_train = y
            self.paths_train = image_paths
            self.x_val = np.array([])
            self.y_val = np.array([])
            self.paths_val = []
            self.x_test = np.array([])
            self.y_test = np.array([])
            self.paths_test = []
            print(f"Train: {len(self.x_train)}, Val: 0, Test: 0")
            return
        else:
            # equal test and val size
            val_size = test_size
            trainval_size = 1.0 - test_size

            # First split: train+val and test
            x_trainval, self.x_test, y_trainval, self.y_test, paths_trainval, self.paths_test = train_test_split(
                x, y, image_paths, test_size=test_size, random_state=random_state)

            # Second split: train and val (val is val_size of the original dataset, so relative to trainval it's val_size/trainval_size)
            rel_val_size = val_size / trainval_size

            self.x_train, self.x_val, self.y_train, self.y_val, self.paths_train, self.paths_val = train_test_split(
                x_trainval, y_trainval, paths_trainval, test_size=rel_val_size, random_state=random_state)

            print(f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test: {len(self.x_test)}")

    def split_dataset_manual(self, test_size=0.2, reshape=False, random_state=None):
        """
        Manually splits the dataset:
        - Test set: rows where 'Image_Path' contains 'Tom'
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
        test_indices = [i for i, p in enumerate(paths_np) if "tom" in p]
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
            x_trainval, y_trainval, paths_trainval, test_size=test_size, random_state=0)

        print(f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test (Tom): {len(self.x_test)}")

    def split_dataset_by_fixed_test_group(self, reshape=False, test_group_value=5, group_column='group', test_size=0.2, random_state=0):
        # Copy dataframe
        x = self.df.copy()

        # Sort by Pose_Class from P1 to P10
        x['Pose_Class'] = pd.Categorical(x['Pose_Class'], categories=[f'P{i}' for i in range(1, 11)], ordered=True)
        x = x.sort_values('Pose_Class')
        x['Pose_Class'], uniques = pd.factorize(x['Pose_Class'])
        
        # Extract groups (e.g., person names)
        groups = x[group_column]
        print(groups)
        
        # Boolean masks for test and trainval
        test_mask = groups == int(test_group_value)
        trainval_mask = ~test_mask

        # Split dataframes
        df_test = x[test_mask]
        df_trainval = x[trainval_mask]

        # Split trainval into train and val randomly
        train_idx, val_idx = train_test_split(
            df_trainval.index, test_size=test_size, random_state=random_state, shuffle=True
        )

        df_train = df_trainval.loc[train_idx]
        df_val = df_trainval.loc[val_idx]

        # Helper to convert df to arrays
        def df_to_xy(df):
            df = df.copy()  # To avoid modifying original

            # Extract and remove metadata columns
            image_paths = df.pop('Image_Path').to_list()
            pose_labels = df.pop('Pose_Class')
            groups = df.pop(group_column).to_list()

            y = keras.utils.to_categorical(pose_labels)

            # Feature data
            x_data = df.astype('float64').to_numpy()

            # Reshape if needed
            if reshape:
                print(f"Before reshape: {x_data.shape}")
                x_data = self.reshape_keypoints(x_data)
                print(f"After reshape: {x_data.shape}")

            return x_data, y, image_paths, groups

        self.x_train, self.y_train, self.paths_train, self.groups_train = df_to_xy(df_train)
        self.x_val, self.y_val, self.paths_val, self.groups_val = df_to_xy(df_val)
        self.x_test, self.y_test, self.paths_test, self.groups_test = df_to_xy(df_test)

        # Save groups as numpy arrays for compatibility with scikit-learn splitters
        self.groups_train = np.array(self.groups_train)
        self.groups_val = np.array(self.groups_val)
        self.groups_test = np.array(self.groups_test)

        print(f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test (fixed group '{test_group_value}'): {len(self.x_test)}")

    
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