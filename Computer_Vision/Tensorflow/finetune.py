import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
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

# Load the saved model with custom objects
model = tf.keras.models.load_model("weights_swing.best.SavedModel", custom_objects={"landmarks_to_embedding": landmarks_to_embedding})

# (Optional) Freeze some layers if you do not want to update all layers during fine-tuning
# for layer in model.layers[:-2]:  # Freezing all layers except the last two
#     layer.trainable = False

# Compile the model again if needed
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load new data
csvs_out_train_new_path = 'output/DOWNTHELINE_train_Swing_data.csv'
csvs_out_test_new_path = 'output/DOWNTHELINE_test_Swing_data.csv'

# Load the new train data
X_new_train, y_new_train, class_names, _ = load_pose_landmarks(csvs_out_train_new_path)
X_new_train, X_new_val, y_new_train, y_new_val = train_test_split(X_new_train, y_new_train, test_size=0.15)

# Load the new test data
X_new_test, y_new_test, _, df_new_test = load_pose_landmarks(csvs_out_test_new_path)

# Add a checkpoint callback to store the checkpoint that has the highest validation accuracy during fine-tuning.
checkpoint_path_finetune = "weights_swing_finetune.best.SavedModel"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_finetune,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

# Fine-tune the model on the new data
history_finetune = model.fit(X_new_train, y_new_train,
                             epochs=100,
                             batch_size=16,
                             validation_data=(X_new_val, y_new_val),
                             callbacks=[checkpoint, earlystopping])

# Evaluate the fine-tuned model using the new test dataset
loss_finetune, accuracy_finetune = model.evaluate(X_new_test, y_new_test)

print('Fine-tuned model accuracy:', accuracy_finetune)

# Save the fine-tuned model in SavedModel format
model.save("saved_model/dtl_swing_model_finetuned")

# Convert the fine-tuned model to TensorFlow Lite
@tf.function
def model_fn(x):
    return model(x)

# Get a concrete function from the Keras model
concrete_func = model_fn.get_concrete_function(tf.TensorSpec(shape=[None, 51], dtype=tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_finetuned = converter.convert()

print('Fine-tuned model size: %dKB' % (len(tflite_model_finetuned) / 1024))

with open('dtl_swing_classifier_finetuned.tflite', 'wb') as f:
    f.write(tflite_model_finetuned)

with open('dtl_swing_labels.txt', 'w') as f:
    f.write('\n'.join(str(class_name) for class_name in class_names))
