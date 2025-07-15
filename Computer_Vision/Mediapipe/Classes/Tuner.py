import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import keras_tuner as kt

class CVTuner(kt.BayesianOptimization):
    """Custom Keras Tuner that uses cross-validation for hyperparameter evaluation"""
    
    def __init__(self, hypermodel, objective, cv_folds=5, **kwargs):
        super().__init__(hypermodel, objective, **kwargs)
        self.cv_folds = cv_folds
        self.trial_metrics = {}  # Store custom metrics
    
    def run_trial(self, trial, X, y, epochs=200, **kwargs):
        """
        Run a single trial with cross-validation
        """
        hp = trial.hyperparameters
        
        # Use StratifiedKFold for classification to maintain class distribution
        if len(np.unique(y)) > 1:
            print("Using Stratified KFold")
            kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_losses = []
        cv_precisions = []
        cv_recalls = []


        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"  Fold {fold + 1}/{self.cv_folds}")
            
            # Split data for this fold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build fresh model for this fold
            model = self.hypermodel.build(hp)
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                verbose=0
            )
            
            # Train model
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0  # Reduce verbosity
            )
            print(f"{len(history.epoch)} epoch completed.")

            # Evaluate on validation set
            val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(val_accuracy)
            cv_losses.append(val_loss)
            
            # Evaluate predictions
            y_pred = model.predict(X_val_fold)
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            precision = precision_score(y_val_fold, y_pred_labels, average='macro', zero_division=0)
            recall = recall_score(y_val_fold, y_pred_labels, average='macro', zero_division=0)
            
            cv_precisions.append(precision)
            cv_recalls.append(recall)

            # Clean up to save memory
            del model
            tf.keras.backend.clear_session()
        
        # Calculate mean CV score
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        mean_cv_loss = np.mean(cv_losses)
        mean_precision = np.mean(cv_precisions)
        mean_recall = np.mean(cv_recalls)

        # Store additional metrics in class attribute
        self.trial_metrics[trial.trial_id] = {
            'std': std_cv_score,
            'precision': mean_precision,
            'recall': mean_recall,
            'loss': mean_cv_loss
        }

        print(f"  CV Score: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
        
        # Return the metric that the tuner is optimizing for
        return mean_cv_score