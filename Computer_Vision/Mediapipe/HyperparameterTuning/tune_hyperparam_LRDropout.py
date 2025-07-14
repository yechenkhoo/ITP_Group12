"""
Hyperparameter tuning for fixed architectures using cross-validation.
Uses DeepLearningModel.tune_hyperparameters()
"""

from Classes.DeepLearningModel import DeepLearningModel
from Classes.PoseDataset import PoseDataset
import pandas as pd
import os

def main():
    path_csv = "dataset3.csv"
    test_run_name = "hyperparameter_tuning"
    folder = f"output/{test_run_name}"
    os.makedirs(folder, exist_ok=True)
    
    # Load dataset
    print("[INFO] Loading dataset...")
    data = PoseDataset(path_csv)
    data.load_csv_data()
    

    # Tune learning rate (1e-4 to 1e-2) only:
    # - 'mlp_basic': Simple MLP without dropout
    # - 'cnn_3_block': 3-block CNN without dropout
    
    # Tune learning rate + dropout:
    # - 'mlp_basic_tunable': MLP with tunable dropout (0.1-0.7)
    # - 'cnn_3_block_tunable': CNN with tunable dropout (0.1-0.7)
    architecture = 'mlp_basic_tunable'
    
    print(f"Architecture: {architecture}")
    print("Strategy: 5-fold CV on train+val, test set remains unseen")
    
    # IMPORTANT: Data reshaping for CNN architectures
    if 'cnn' in architecture:
        print("[INFO] CNN architecture detected - data will be reshaped to (33, 4)")
        # CNN requires reshaped data: (samples, 33, 4) instead of (samples, 132)
        data.split_dataset(test_size=0.2, reshape=True, random_state=42)  # reshape=True for CNN
    else:
        print("[INFO] MLP architecture detected - using flattened data (132 features)")
        # MLP uses flattened data: (samples, 132)
        data.split_dataset(test_size=0.2, reshape=False, random_state=42)  # reshape=False for MLP
    
    print(f"Dataset split:")
    print(f"  Train: {data.x_train.shape[0]} samples")
    print(f"  Val: {data.x_val.shape[0]} samples") 
    print(f"  Test: {data.x_test.shape[0]} samples (UNSEEN)")
    print(f"  Classes: {data.allClasses}")
    
    # Initialize model
    model = DeepLearningModel(
        input_shape=data.x_train.shape[1],  # Flattened: 132 features
        class_count=data.classCount,
        checkpoint_path=f"{folder}/best_model.h5",
        name="pose_classifier"
    )
    
    # Run hyperparameter tuning with CV
    results = model.tune_hyperparameters(
        data=data,
        architecture_name=architecture,
        k_folds=5,
        max_trials=20,
        epochs=30
    )
    
    # Display and save results
    print(f"\n[INFO] Tuning complete!")
    print(f"Best hyperparameters: {results['best_hyperparameters']}")
    print(f"Best CV score: {results['best_cv_score']:.4f}")
    
    # Save detailed results
    trials_df = pd.DataFrame(results['all_trials'])
    results_file = f"{folder}/tuning_results_{architecture}.csv"
    trials_df.to_csv(results_file, index=False)
    print(f"[INFO] Results saved: {results_file}")
    
    # Final evaluation on unseen test set
    print(f"\n" + "="*60)
    print("FINAL EVALUATION ON UNSEEN TEST SET")
    print("="*60)
    
    # Convert test labels if needed
    if len(data.y_test.shape) > 1 and data.y_test.shape[1] > 1:
        y_test_sparse = data.y_test.argmax(axis=1)
    else:
        y_test_sparse = data.y_test
    
    test_loss, test_accuracy = model.model.evaluate(data.x_test, y_test_sparse, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Note: This is your FINAL performance estimate!")
    
    print(f"\n[INFO] Summary:")
    print(f"  Architecture: {architecture}")
    print(f"  Best CV score: {results['best_cv_score']:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Best hyperparameters: {results['best_hyperparameters']}")

if __name__ == "__main__":
    main()