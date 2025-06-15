import seaborn as sns
import keras
import pandas as pd
from keras import layers, Sequential
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
        The validation set will be 20% of the non-test set.
        e.g. if test_size is 0.2, training set is 0.64 and validation set is 0.16.

        - reshape: reshapes the data into 33x4 instead of 132x1
        """
        x = self.df.copy()
        y, _ = x.pop('Pose_Class').factorize()
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Converts it back to 33x4 instead of 132x1
        if reshape:
            x_np = x.to_numpy()
            x = self.reshape_keypoints(x_np)

        # Split full dataset into train+validation set and test set
        x_trainval, self.x_test, y_trainval, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        # Split train+validation into train set and val set
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_trainval, y_trainval, test_size=test_size, random_state=random_state)
    
    def reshape_keypoints(self, X_flat):
        """
        # Converts a flat input like (132,) to a structured (33, 4) shape.
        # Assumes 33 keypoints each with 4 features: x, y, z, visibility.
        # Works for a single sample or batch of samples.
        """
        if X_flat.ndim == 1:
            return X_flat.reshape((33, 4))
        elif X_flat.ndim == 2:
            return X_flat.reshape((-1, 33, 4))
        else:
            raise ValueError("Input must be a 1D or 2D array")