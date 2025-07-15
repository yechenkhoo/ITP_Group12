"""
Hyperparameter tuning for fixed architectures using cross-validation.


Set your 
Command to run in Mediapipe/:
python -m HyperparameterTuning.tune_hyperparam_main
"""

from Classes.DeepLearningModel import DeepLearningModel
from Classes.PoseDataset import PoseDataset
import pandas as pd
import os
import json
from sklearn.metrics import precision_score, recall_score

def main():
    architectures = ['mlp', 'cnn']

    for architecture in architectures:
        print(f"\nArchitecture: {architecture}")
        path_csv = "dataset3.csv"
        test_run_name = "hyperparameter_tuning_kerastuner_bayesian"
        folder = f"output/{test_run_name}/{architecture}"
        os.makedirs(folder, exist_ok=True)
        
        # Load dataset
        print("[INFO] Loading dataset...")
        data = PoseDataset(path_csv)
        data.load_csv_data()
        
        # Define architecture to tune
        print("Strategy: 5-fold CV on train+val, test set remains unseen")
        
        # Reshape if CNN is selected
        if 'cnn' in architecture:
            print("[INFO] CNN architecture detected - reshaping input to (33, 4)")
            data.split_dataset(test_size=0.2, reshape=True, random_state=42)
        else:
            print("[INFO] MLP architecture detected - using flattened input (132 features)")
            data.split_dataset(test_size=0.2, reshape=False, random_state=42)
        
        print("\n[INFO] Dataset Summary:")
        print(f"  Train samples: {data.x_train.shape[0]}")
        print(f"  Val samples:   {data.x_val.shape[0]}")
        print(f"  Test samples:  {data.x_test.shape[0]} (UNSEEN)")
        print(f"  Classes:       {data.allClasses}")
        
        # Initialize model
        model = DeepLearningModel(
            input_shape= (data.x_train.shape[1],) if architecture=='mlp' else tuple(data.x_train.shape[1:]),
            class_count=data.classCount,
            checkpoint_path=f"{folder}/best_model.keras",
            name=f"{test_run_name}"
        )
        
        # Run hyperparameter tuning
        results = model.tune_hyperparameters_kerastuner(
            directory=folder,
            data=data,
            architecture_name=architecture,
            max_trials=30,
            epochs=200,
            cv_folds=5
        )
        
        # # Run hyperparameter tuning
        # results = model.tune_hyperparameters_archi(
        #     data=data,
        #     architecture_name=architecture,
        #     k_folds=5,
        #     max_trials=50,
        #     epochs=200,
        # )
        

        # Save and report results
        print("\n[INFO] Tuning complete!")
        print(f"Best hyperparameters: {results['best_hyperparameters']}")
        print(f"Best score:        {results.get('best_cv_score', results.get('best_score', None)):.4f}")
        
        # Naming convention for outputs
        best_model_path = f"{folder}/best_model.keras"
        best_results_path = f"{folder}/best_results.json"
        trials_path = f"{folder}/tuning_results.csv"
        
        # Save best model
        model.model.save(best_model_path)
        print(f"[INFO] Best model saved to: {best_model_path}")
        
        # Save best results as JSON
        best_results = {
            "architecture": architecture,
            "best_hyperparameters": results['best_hyperparameters'],
            "best_score": results.get('best_cv_score', None),
            "all_trials": results['all_trials']
        }
        with open(best_results_path, "w") as f:
            json.dump(best_results, f, indent=2)
        print(f"[INFO] Best results saved to: {best_results_path}")
        
        # Save all trials
        trials_df = pd.DataFrame(results['all_trials'])
        trials_df.to_csv(trials_path, index=False)
        print(f"[INFO] Trial results saved to: {trials_path}")
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON UNSEEN TEST SET")
        print("=" * 60)

        if len(data.y_test.shape) > 1 and data.y_test.shape[1] > 1:
            y_test_sparse = data.y_test.argmax(axis=1)
        else:
            y_test_sparse = data.y_test

        test_loss, test_acc = model.model.evaluate(data.x_test, y_test_sparse, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        print("Note: This is your FINAL performance estimate!")
        
        # Compute precision and recall on test set
        y_pred_probs = model.model.predict(data.x_test)
        y_pred_classes = y_pred_probs.argmax(axis=1)
        test_prec = precision_score(y_test_sparse, y_pred_classes, average='macro', zero_division=0)
        test_rec = recall_score(y_test_sparse, y_pred_classes, average='macro', zero_division=0)
        
        # Save final evaluation to CSV
        final_eval_path = f"{folder}/final_test_evaluation.csv"
        final_eval_df = pd.DataFrame([
            {
                "architecture": architecture,
                "best_cv_score": results.get('best_cv_score', results.get('best_score', None)),
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "best_hyperparameters": results['best_hyperparameters']
            }
        ])
        final_eval_df.to_csv(final_eval_path, index=False)
        print(f"[INFO] Final evaluation saved to: {final_eval_path}")

        # Summary
        print("\n[INFO] Summary:")
        print(f"  Architecture:        {architecture}")
        print(f"  Best score:       {results['best_cv_score']:.4f}")
        print(f"  Test set accuracy:   {test_acc:.4f}")
        print(f"  Test set precision:  {test_prec:.4f}")
        print(f"  Test set recall:     {test_rec:.4f}")
        print(f"  Best hyperparameters: {results['best_hyperparameters']}")

if __name__ == "__main__":
    main()
