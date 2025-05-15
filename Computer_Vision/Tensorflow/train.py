import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data import BodyPart

def load_pose_landmarks(csv_path):
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()
    df_to_process.drop(columns=['file_name'], inplace=True)
    classes = df_to_process.pop('class_name').unique()
    y = df_to_process.pop('class_no')
    X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)
    return X, y, classes, dataframe

csvs_out_train_path = 'dtl_train_Swing_data.csv'
csvs_out_test_path = 'dtl_test_Swing_data.csv'

# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)

def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    pose_center_new = tf.broadcast_to(pose_center_new, [tf.shape(landmarks)[0], 17, 2])
    d = landmarks - pose_center_new
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=2))
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size

def normalize_pose_landmarks(landmarks):
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    pose_center = tf.broadcast_to(pose_center, [tf.shape(landmarks)[0], 17, 2])
    landmarks = landmarks - pose_center
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks

def landmarks_to_embedding(landmarks_and_scores):
    reshaped_inputs = tf.reshape(landmarks_and_scores, [-1, 17, 3])
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    embedding = tf.reshape(landmarks, [-1, 34])
    return embedding

# Define the model
inputs = tf.keras.Input(shape=(51,))
embedding = Lambda(lambda x: landmarks_to_embedding(x))(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights_swing.best.SavedModel"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

# Start training
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, earlystopping])

# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(X_test, y_test)

# model.save("model2.keras")

# Use tf.function to create a concrete function
@tf.function
def model_fn(x):
    return model(x)

# Get a concrete function from the Keras model
concrete_func = model_fn.get_concrete_function(tf.TensorSpec(shape=[None, 51], dtype=tf.float32))

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print('Model size: %dKB' % (len(tflite_model) / 1024))

with open('dtl_swing_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

with open('dtl_swing_labels.txt', 'w') as f:
    print("Class Name:", class_names)
    f.write('\n'.join(str(class_names)))
