import seaborn as sns
import keras
import pandas as pd
from keras import layers, Sequential
import argparse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import os

        
class DeepLearningModel:
    def __init__(self, input_shape, class_count, checkpoint_path):
        self.inputShape = input_shape
        self.classCount = class_count
        self.checkpointPath = checkpoint_path
        self.callbacks = []
        self.model = None
        self.history = None


    def build_model(self, model_fn=None):
        if not model_fn:
            print("[INFO] Using default model")
            self.model = Sequential([
                layers.Dense(512, activation='relu', input_shape=[self.inputShape]),
                layers.Dense(256, activation='relu'),
                layers.Dense(self.classCount, activation='softmax')
            ])
        else:
            print("[INFO] Using custom model")
            if callable(model_fn):
                self.model = model_fn(self.inputShape, self.classCount)
            else:
                raise ValueError("model_fn must be a callable function that accepts (inputShape, classCount)")

        print('[INFO] Model architecture built.')
        print(self.model.summary())


    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print('[INFO] Model compiled.')



    def add_callbacks(self, additional_callbacks=[]):
        # Default allback to save best model
        checkpoint = keras.callbacks.ModelCheckpoint(
            self.checkpointPath,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        self.callbacks.append(checkpoint)
        self.callbacks.extend(additional_callbacks)


    def train(self, data, epochs=200, batch_size=16):
        print('[INFO] Model training started...')
        self.history = self.model.fit(
            data.x_train, data.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data.x_val, data.y_val),
            callbacks=self.callbacks
        )
        print(f'[INFO] Model training completed. Best model saved to: {self.checkpointPath}')


    def plot_training_metrics(self, path_to_save):
        if not self.history:
            print('[WARNING] No training history to plot.')
            return

        loss, val_loss, accuracy, val_accuracy = (
            self.history.history['loss'],
            self.history.history['val_loss'],
            self.history.history['accuracy'],
            self.history.history['val_accuracy'],
        )
        epochs = range(len(loss))

        # Plot Graph
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, loss, 'blue', label='loss')
        plt.plot(epochs, val_loss, 'red', label='val_loss')
        plt.plot(epochs, accuracy, 'blue', label='accuracy')
        plt.plot(epochs, val_accuracy, 'green', label='val_accuracy')
        plt.title(str("Model Metrics"))
        plt.legend()

        plot_png = os.path.exists(path_to_save +'_trainingMetrics.png')
        if plot_png:
            os.remove(path_to_save+'_trainingMetrics.png')
            plt.savefig(path_to_save+'_trainingMetrics.png', bbox_inches='tight')
        else:
            plt.savefig(path_to_save+'_trainingMetrics.png', bbox_inches='tight')
        print('[INFO] Successfully Saved metrics.png')


    def plot_confusion_matrix(self, data, path_to_save, dataset="val"):
        if dataset == "val":
            y_pred = self.model.predict(data.x_val)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = data.y_val.argmax(axis=1)
        elif dataset == "test":
            y_pred = self.model.predict(data.x_test)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = data.y_test.argmax(axis=1)
        else:
            print('[WARNING] Please choose from "val", or "test".')

        cm = confusion_matrix(y_true, y_pred_classes)
        acc = accuracy_score(y_true, y_pred_classes)
        prec = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.allClasses, yticklabels=data.allClasses)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"{dataset.capitalize()} Confusion Matrix\n"
                f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        filename = f"{path_to_save}_{dataset}_confusion_matrix.png"
        confusion_matrix_png = os.path.exists(filename)
        if confusion_matrix_png:
            os.remove(filename)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(filename, bbox_inches='tight')
        print(f'[INFO] Successfully Saved Confusion Matrix for {dataset} set as {filename}')