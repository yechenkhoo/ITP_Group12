import seaborn as sns
import keras
import pandas as pd
from keras import layers, Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import tensorflow as tf
from .ModelFactory import ModelFactory
from .Tuner import CVTuner
import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras import optimizers

class DeepLearningModel:
    def __init__(self, input_shape, class_count, checkpoint_path, name):
        self.inputShape = input_shape
        self.classCount = class_count
        self.checkpointPath = checkpoint_path
        self.callbacks = []
        self.model = None
        self.name = name
        self.history = None
        self.valResults = []
        self.testResults = []


    def build_model(self, model_fn=None):
        if not model_fn:
            print("[INFO] Using default model")
            self.model = Sequential([
                layers.Dense(512, activation='relu', input_shape=self.inputShape),
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

    def build_model_with_hp(self, hp, architecture):
        model = Sequential()
        model.add(layers.Input(shape=self.inputShape))

        if architecture=="mlp":
            model.add(layers.Dense(
                hp.Int("units1", 128, 512, step=64),
                activation='relu',
            ))
            model.add(layers.Dropout(hp.Float("dropout1", 0.1, 0.7, step=0.1)))
            model.add(layers.Dense(
                hp.Int("units2", 64, 256, step=64),
                activation='relu'
            ))
            model.add(layers.Dropout(hp.Float("dropout2", 0.1, 0.7, step=0.1)))
            model.add(layers.Dense(self.classCount, activation='softmax'))

        else:
            model.add(layers.Conv1D(
                hp.Int("filters1", 16, 64, step=16),
                hp.Choice("kernel_size", [3, 5]),
                activation='relu',
                padding='same'
            ))
            model.add(layers.MaxPooling1D(hp.Choice("pool_size", [2, 3])))
            model.add(layers.Conv1D(
                hp.Int("filters2", 32, 128, step=32),
                hp.Choice("kernel_size", [3, 5]),
                activation='relu',
                padding='same'
            ))
            model.add(layers.MaxPooling1D(hp.Choice("pool_size", [2, 3])))
            model.add(layers.Conv1D(
                hp.Int("filters3", 32, 128, step=32),
                hp.Choice("kernel_size", [3, 5]),
                activation='relu',
                padding='same'
            ))
            model.add(layers.GlobalMaxPooling1D())
            dense_units = hp.Int("dense_units", 64, 512, step=64)
            model.add(layers.Dense(dense_units, activation='relu'))
            dropout = hp.Float("dropout", 0.1, 0.7, step=0.1)
            model.add(layers.Dropout(dropout))
            model.add(layers.Dense(self.classCount, activation='softmax'))


        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling='log')
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def build_model_with_hp_layertuning(self, hp, architecture):
        model = Sequential()
        model.add(layers.Input(shape=self.inputShape))

        if architecture == "mlp":
            num_layers = hp.Int("num_dense_layers", 2, 5)
            for i in range(num_layers):
                units = hp.Int(f"units_{i+1}", 64, 512, step=64)
                model.add(layers.Dense(units, activation='relu'))
                dropout = hp.Float(f"dropout_{i+1}", 0.1, 0.7, step=0.1)
                model.add(layers.Dropout(dropout))
            model.add(layers.Dense(self.classCount, activation='softmax'))
        else:
            num_blocks = hp.Int("num_blocks", 1, 3)
            num_layers_per_block = hp.Int("num_layers_per_block", 1, 3)
            for b in range(num_blocks):
                for l in range(num_layers_per_block):
                    filters = hp.Int(f"filters_b{b+1}_l{l+1}", 16, 128, step=16)
                    kernel_size = hp.Choice(f"kernel_size_b{b+1}_l{l+1}", [3, 5])
                    model.add(layers.Conv1D(filters, kernel_size, activation='relu', padding='same'))
                pool_size = hp.Choice(f"pool_size_b{b+1}", [2, 3])
                model.add(layers.MaxPooling1D(pool_size))
            model.add(layers.GlobalMaxPooling1D())
            dense_units = hp.Int("dense_units", 64, 512, step=64)
            model.add(layers.Dense(dense_units, activation='relu'))
            dropout = hp.Float("dropout", 0.1, 0.7, step=0.1)
            model.add(layers.Dropout(dropout))
            model.add(layers.Dense(self.classCount, activation='softmax'))


        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling='log')
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    

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

        # Create folder called path_to_save
        os.makedirs(path_to_save, exist_ok=True)
        path = f"{path_to_save}/{self.name}_trainingMetrics.png"
        plt.savefig(path, bbox_inches='tight')
        print('[INFO] Successfully Saved metrics.png')


    def plot_confusion_matrix(self, data, path_to_save, dataset="val"):
        if dataset == "val":
            y_pred = self.model.predict(data.x_val)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = data.y_val.argmax(axis=1)
            x_data = data.x_val
            img_paths = data.paths_val
            getattr(data, 'img_paths_val', None)  # Get image paths if available
        elif dataset == "test":
            y_pred = self.model.predict(data.x_test)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = data.y_test.argmax(axis=1)
            x_data = data.x_test
            img_paths = data.paths_test
        else:
            print('[WARNING] Please choose from "val", or "test".')
            return

        cm = confusion_matrix(y_true, y_pred_classes)
        acc = accuracy_score(y_true, y_pred_classes)
        prec = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)

        if dataset == "val":
            self.valResults.extend((acc, prec, rec))
        elif dataset == "test":
            self.testResults.extend((acc, prec, rec))
        
        all_records = []
        for i in range(len(y_true)):
            # Get confidence (probability of predicted class)
            confidence = y_pred[i][y_pred_classes[i]]
            
            # Get image path if available
            img_path = img_paths[i] if img_paths is not None else f"sample_{i}"
            
            if hasattr(x_data, 'iloc'):  # pandas DataFrame
                landmarks_flat = x_data.iloc[i].values.tolist()
            elif isinstance(x_data, np.ndarray):  # numpy array
                if x_data.ndim == 3:  # Shape: [samples, keypoints, features] - if reshaped
                    landmarks_flat = x_data[i].flatten().tolist()
                else:  # Shape: [samples, features] - if not reshaped (132 features)
                    landmarks_flat = x_data[i].tolist()
            else:
                landmarks_flat = []
            
            # Create record
            record = {
                "img_path": img_path,
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred_classes[i]),
                "confidence": float(confidence),
                "landmarks": ' '.join(map(str, landmarks_flat))
            }
            all_records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(all_records)
        path = f"{path_to_save}/{self.name}"
        csv_path = f"{path}_{dataset}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved prediction CSV with confidence at {csv_path}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.allClasses, yticklabels=data.allClasses)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"{dataset.capitalize()} Confusion Matrix\n"
                f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        filename = f"{path}_{dataset}_confusion_matrix.png"
        confusion_matrix_png = os.path.exists(filename)
        if confusion_matrix_png:
            os.remove(filename)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(filename, bbox_inches='tight')
        print(f'[INFO] Successfully Saved Confusion Matrix for {dataset} set as {filename}')


    def tune_hyperparameters_kerastuner(self, directory, data, architecture_name, max_trials=20, epochs=200, cv_folds=5):
        """
        Hyperparameter tuning using KerasTuner with Cross-Validation.
        Args:
            data: PoseDataset object
            architecture_name: ModelFactory architecture name
            max_trials: Number of hyperparameter sets to try
            epochs: Training epochs per trial
            cv_folds: Number of cross-validation folds
        Returns:
            dict: Results with best hyperparameters and scores
        """
        print(f"[INFO] Hyperparameter tuning (KerasTuner) with {cv_folds} for {architecture_name}")
        print(f"Max trials: {max_trials}, Epochs per trial: {epochs}")

        # Convert one-hot to sparse if needed
        if len(data.y_train.shape) > 1 and data.y_train.shape[1] > 1:
            y_train_sparse = np.argmax(data.y_train, axis=1)
        else:
            y_train_sparse = data.y_train

        if len(data.y_val.shape) > 1 and data.y_val.shape[1] > 1:
            y_val_sparse = np.argmax(data.y_val, axis=1)
        else:
            y_val_sparse = data.y_val

        # Combine train and validation sets for CV
        X_full = np.concatenate([data.x_train, data.x_val])
        y_full = np.concatenate([y_train_sparse, y_val_sparse])

        def build_wrapper(hp):
            return self.build_model_with_hp_layertuning(hp, architecture=architecture_name)

        tuner = CVTuner(
            build_wrapper,
            objective=kt.Objective('score', direction='max'),
            cv_folds=cv_folds,
            max_trials=max_trials,
            executions_per_trial=1,
            directory=directory,
            project_name=f"{architecture_name}_cv"
        )

        tuner.search(
            X_full, y_full,
            epochs=epochs
        )

        best_trial = tuner.oracle.get_best_trials(1)[0]
        hp = best_trial.hyperparameters
        final_model = self.build_model_with_hp(hp, architecture=architecture_name)

        # Use train and val data separately for final fit
        final_model.fit(
            data.x_train,
            y_train_sparse,
            epochs=500,
            validation_data=(data.x_val, y_val_sparse),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
            verbose=1
        )
        self.model = final_model

        # Collect trial results
        all_trials = []
        for trial in tuner.oracle.get_best_trials(num_trials=max_trials):
            # Get custom metrics from the tuner's trial_metrics dictionary
            custom_metrics = tuner.trial_metrics.get(trial.trial_id, {})
            
            trial_data = {
                **trial.hyperparameters.values,
                "cv_score": trial.score,
                "cv_std": custom_metrics.get("std", 0),
                "cv_prec": custom_metrics.get("precision", 0),
                "cv_recall": custom_metrics.get("recall", 0),
            }
            all_trials.append(trial_data)
        # Return results
        return {
            "best_hyperparameters": best_trial.hyperparameters.values,
            "best_cv_score": best_trial.score,
            "cv_folds": cv_folds,
            "all_trials": all_trials
        }

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj