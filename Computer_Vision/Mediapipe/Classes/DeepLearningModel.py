import seaborn as sns
import keras
import pandas as pd
from keras import layers, Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping


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

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']
        epochs = range(1, len(loss) + 1)

        # Plot Graph
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, loss, 'blue', label='loss')
        plt.plot(epochs, val_loss, 'red', label='val_loss')
        plt.plot(epochs, accuracy, 'blue', linestyle='dashed', label='accuracy')
        plt.plot(epochs, val_accuracy, 'green', linestyle='dashed', label='val_accuracy')
        plt.title("Model Metrics")
        plt.xlabel("Epoch")
        plt.legend()

        # Create folder called path_to_save
        os.makedirs(path_to_save, exist_ok=True)
        img_path = f"{path_to_save}/{self.name}_trainingMetrics.png"
        plt.savefig(img_path, bbox_inches='tight')
        print(f'[INFO] Successfully Saved metrics plot as {img_path}')

        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'loss': loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy
        })
        csv_path = f"{path_to_save}/{self.name}_trainingMetrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f'[INFO] Successfully Saved metrics CSV as {csv_path}')


    def load_best_model(self):
        # Load the best model saved during training
        if os.path.exists(self.checkpointPath):
            self.model = keras.models.load_model(self.checkpointPath)
            print(f'[INFO] Loaded best model from {self.checkpointPath}')
        else:
            print(f'[WARNING] No saved model found at {self.checkpointPath}')


    def plot_confusion_matrix(self, data, path_to_save, dataset="val"):
        # Load the best model for final evaluation
        self.load_best_model()
        
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
        fscore = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # Store results properly
        if dataset == "val":
            self.valResults = [acc, prec, rec, fscore]
        elif dataset == "test":
            self.testResults = [acc, prec, rec, fscore]

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

    def cross_validate(self, data, log_dir, k_folds=5, epochs=100, batch_size=16):
        """
        Perform k-fold cross validation to understand training data quality.
        Logs results for each fold to CSV files.
        Returns: dict with fold results and statistics
        """

        # Prepare log directory
        os.makedirs(log_dir, exist_ok=True)

        # Get original labels (non-one-hot)
        y_labels = data.y_train.argmax(axis=1)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = {
            'fold_accuracies': [],
            'fold_losses': [],
            'fold_f1_scores': [],
            'fold_precisions': [],
            'fold_recalls': [],
            'mean_accuracy': 0,
            'std_accuracy': 0,
            'mean_f1': 0,
            'std_f1': 0,
            'mean_precision': 0,
            'std_precision': 0,
            'mean_recall': 0,
            'std_recall': 0,
            'all_fold_histories': [],
            'epochs_per_fold': []
        }

        print(f"[INFO] Starting {k_folds}-fold cross validation...")

        # Prepare summary DataFrame outside the loop to avoid overlap
        summary_path = os.path.join(log_dir, "cv_summary.csv")
        df_summary = pd.DataFrame(columns=['fold', 'val_accuracy', 'val_loss', 'val_f1_score', 'val_precision', 'val_recall', 'epochs_completed'])

        # Ensure x_train and y_train are numpy arrays for indexing
        x_train_np = data.x_train.values if hasattr(data.x_train, 'values') else np.array(data.x_train)
        y_train_np = data.y_train.values if hasattr(data.y_train, 'values') else np.array(data.y_train)

        # Split using K-Fold
        splits = list(skf.split(x_train_np, y_labels))
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n[INFO] Training Fold {fold + 1}/{k_folds}")

            # Split data for this fold
            x_fold_train, x_fold_val = x_train_np[train_idx], x_train_np[val_idx]
            y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

            # Create fresh model for this fold
            temp_model = keras.models.clone_model(self.model)
            temp_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train on this fold
            history = temp_model.fit(
                x_fold_train, y_fold_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_fold_val, y_fold_val),
                verbose=0,
                callbacks=[early_stop]
            )

            # Number of epochs completed for this fold
            epochs_completed = len(history.history['loss'])
            cv_results['epochs_per_fold'].append(epochs_completed)

            # Evaluate this fold
            val_loss, val_acc = temp_model.evaluate(x_fold_val, y_fold_val, verbose=0)
            
            # Calculate F1 score for this fold
            y_pred = temp_model.predict(x_fold_val, verbose=0)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true_classes = y_fold_val.argmax(axis=1)
            val_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            
            # Calculate Precision and Recall for this fold
            val_precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            val_recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            
            cv_results['fold_accuracies'].append(val_acc)
            cv_results['fold_losses'].append(val_loss)
            cv_results['fold_f1_scores'].append(val_f1)
            cv_results['fold_precisions'].append(val_precision)
            cv_results['fold_recalls'].append(val_recall)
            cv_results['all_fold_histories'].append(history.history)

            print(f"Fold {fold + 1} - Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Epochs: {epochs_completed}/{epochs}")

            # Write per-fold log to CSV
            log_path = os.path.join(log_dir, f"fold_{fold+1}_log.csv")
            df_log = pd.DataFrame({
                'epoch': np.arange(1, len(history.history['loss']) + 1),
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            })
            df_log.to_csv(log_path, index=False)

            # Append summary for this fold to the DataFrame
            df_summary = pd.concat([
                df_summary,
                pd.DataFrame([{
                    'fold': fold + 1,
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'val_f1_score': val_f1,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'epochs_completed': epochs_completed
                }])
            ], ignore_index=True)

            df_summary.to_csv(summary_path, index=False)

        # Calculate statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        cv_results['mean_f1'] = np.mean(cv_results['fold_f1_scores'])
        cv_results['std_f1'] = np.std(cv_results['fold_f1_scores'])
        cv_results['mean_precision'] = np.mean(cv_results['fold_precisions'])
        cv_results['std_precision'] = np.std(cv_results['fold_precisions'])
        cv_results['mean_recall'] = np.mean(cv_results['fold_recalls'])
        cv_results['std_recall'] = np.std(cv_results['fold_recalls'])

        print(f"\n[INFO] Cross Validation Results:")
        print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Mean F1 Score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"Mean Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
        print(f"Mean Recall: {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in cv_results['fold_accuracies']]}")
        print(f"Individual Fold F1 Scores: {[f'{f1:.4f}' for f1 in cv_results['fold_f1_scores']]}")
        print(f"Individual Fold Precisions: {[f'{prec:.4f}' for prec in cv_results['fold_precisions']]}")
        print(f"Individual Fold Recalls: {[f'{rec:.4f}' for rec in cv_results['fold_recalls']]}")
        print(f"Epochs per fold: {cv_results['epochs_per_fold']} (out of {epochs})")

        # Log statistics to a file
        stats_path = os.path.join(log_dir, "cv_stats.txt")
        lowest_acc = min(cv_results['fold_accuracies'])
        highest_acc = max(cv_results['fold_accuracies'])
        lowest_f1 = min(cv_results['fold_f1_scores'])
        highest_f1 = max(cv_results['fold_f1_scores'])
        lowest_precision = min(cv_results['fold_precisions'])
        highest_precision = max(cv_results['fold_precisions'])
        lowest_recall = min(cv_results['fold_recalls'])
        highest_recall = max(cv_results['fold_recalls'])

        with open(stats_path, "w") as f:
            f.write(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f}\n")
            f.write(f"Std Accuracy: {cv_results['std_accuracy']:.4f}\n")
            f.write(f"Mean F1 Score: {cv_results['mean_f1']:.4f}\n")
            f.write(f"Std F1 Score: {cv_results['std_f1']:.4f}\n")
            f.write(f"Mean Precision: {cv_results['mean_precision']:.4f}\n")
            f.write(f"Std Precision: {cv_results['std_precision']:.4f}\n")
            f.write(f"Mean Recall: {cv_results['mean_recall']:.4f}\n")
            f.write(f"Std Recall: {cv_results['std_recall']:.4f}\n")
            f.write(f"Fold Accuracies: {cv_results['fold_accuracies']}\n")
            f.write(f"Fold F1 Scores: {cv_results['fold_f1_scores']}\n")
            f.write(f"Fold Precisions: {cv_results['fold_precisions']}\n")
            f.write(f"Fold Recalls: {cv_results['fold_recalls']}\n")
            f.write(f"Epochs per fold: {cv_results['epochs_per_fold']} (out of {epochs})\n")
            f.write(f"Lowest Fold Accuracy: {lowest_acc:.4f}\n")
            f.write(f"Highest Fold Accuracy: {highest_acc:.4f}\n")
            f.write(f"Lowest Fold F1 Score: {lowest_f1:.4f}\n")
            f.write(f"Highest Fold F1 Score: {highest_f1:.4f}\n")
            f.write(f"Lowest Fold Precision: {lowest_precision:.4f}\n")
            f.write(f"Highest Fold Precision: {highest_precision:.4f}\n")
            f.write(f"Lowest Fold Recall: {lowest_recall:.4f}\n")
            f.write(f"Highest Fold Recall: {highest_recall:.4f}\n")

        return cv_results