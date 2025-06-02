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
        self.df = pd.read_csv(self.path)
        self.allClasses = sorted(self.df['Pose_Class'].unique())
        self.classCount = len(self.allClasses)
    
    def split_dataset(self, test_size=0.2, random_state=0):
        x = self.df.copy()
        y, _ = x.pop('Pose_Class').factorize()
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Split full dataset into train+validation set and test set
        x_trainval, self.x_test, y_trainval, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        # Split train+validation into train set and val set
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_trainval, y_trainval, test_size=test_size, random_state=random_state)