import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data import BodyPart

def load_pose_landmarks(csv_path):
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()
    df_to_process.drop(columns=['file_name'], inplace=True)
    classes = df_to_process.pop('class_name').unique()
    y = df_to_process.pop('class_no')
    X = df_to_process.astype('float64')
    y = tf.keras.utils.to_categorical(y)
    return X, y, classes, dataframe

# Load the TFLite model
tflite_model_path = 'dtl_swing_classifier_finetuned.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the test data
csvs_out_test_path = 'output/DOWNTHELINE_test_Swing_data.csv'
X_test, y_test, class_names, df_test = load_pose_landmarks(csvs_out_test_path)

# Ensure the input shape matches
input_shape = input_details[0]['shape']
X_test = np.array(X_test, dtype=np.float32)

# Run inference on the test data
y_pred = []
for i in range(len(X_test)):
    input_data = np.expand_dims(X_test[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(output_data)

y_pred = np.array(y_pred).squeeze()
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Generate a classification report
report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
print(f'Classification Report:\n{report}')

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print(f'Confusion Matrix:\n{conf_matrix}')

# Visualize the confusion matrix using matplotlib (optional)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
