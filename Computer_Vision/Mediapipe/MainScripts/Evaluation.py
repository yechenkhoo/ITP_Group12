"""
Run this using:
python -m MainScripts.Evaluation
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from Classes.PoseDataset import PoseDataset
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, classification_report
import os


def main():
    # CONFIG VARIABLES
    # naming_convention="output/TomSplit/hyperparameter_tuning_kerastuner_" #HPTUNING
    naming_convention="output/D4_tomtest" #BENCHMARKING
    all_models = [
        "MLP_Basic",
        "MLP_Deep",
        "MLP_Dropout",
        "MLP_Attention",
        "CNN_Basic",
        "CNN_Attention",
        "CNN_2D",
        "CNN_3_block",
    ]
    tuning_types=["bayesian", "random"] # HPTUNING
    layer_tunings=["_layertuning", ""] # HPTUNING
    model_architectures=all_models #BENCHMARKING
    # model_architectures=["cnn", "mlp"] # HPTUNING
    DATA_PATH = "dataset4.csv"
    MODEL_PATH = "models/basemodel.keras"

    all_results = []
    all_per_class_results = []
    
    for tuning_type in tuning_types:
        for layer_tuning in layer_tunings:
            for model_architecture in model_architectures:
                model_path = MODEL_PATH # BASELINE
                output_path = "output/BASELINEeval.csv" # BASELINE
                # model_path = f"{naming_convention}/{model_architecture}.h5" # BENCHMARKING
                # output_path = f"{naming_convention}/{model_architecture}/eval.csv" # BENCHMARKING
                # model_path = f"{naming_convention}{tuning_type}{layer_tuning}_tomsplit/{model_architecture}/best_model.keras" # HPTUNING
                # output_path = f"{naming_convention}{tuning_type}{layer_tuning}_tomsplit/{model_architecture}/eval.csv" # HPTUNING
                metrics_path = output_path.replace('.csv', '_metrics.csv')
                per_class_path = output_path.replace('.csv', '_per_class_metrics.csv')
                
                evaluator = Evaluation(
                    model_path=model_path,
                    data_path=DATA_PATH,
                    output_path=output_path,
                    metrics_path=metrics_path,
                    per_class_path=per_class_path,
                    model_architecture=model_architecture
                )
                evaluator.run()
                
                # Load overall metrics
                df = pd.read_csv(metrics_path)
                df['tuning_type'] = tuning_type
                df['layer_tuning'] = layer_tuning if layer_tuning else 'none'
                df['model_architecture'] = model_architecture
                df['model_name'] = f"{tuning_type}_{layer_tuning if layer_tuning else 'none'}_{model_architecture}"
                all_results.append(df)
                
                # Load per-class metrics
                per_class_df = pd.read_csv(per_class_path)
                per_class_df['tuning_type'] = tuning_type
                per_class_df['layer_tuning'] = layer_tuning if layer_tuning else 'none'
                per_class_df['model_architecture'] = model_architecture
                per_class_df['model_name'] = f"{tuning_type}_{layer_tuning if layer_tuning else 'none'}_{model_architecture}"
                all_per_class_results.append(per_class_df)
    
    # # Consolidate and save results
    # if all_results:
    #     master_df = pd.concat(all_results, ignore_index=True)
    #     master_df.to_csv("output/TomSplit/all_evaluations_consolidated.csv", index=False)
    #     print("[INFO] Overall results consolidated to output/TomSplit/all_evaluations_consolidated.csv")
        
    #     # Create summary comparison table
    #     summary_df = master_df[['model_name', 'accuracy', 'precision', 'recall']].copy()
    #     summary_df = summary_df.sort_values('accuracy', ascending=False)
    #     summary_df.to_csv("output/TomSplit/model_comparison_summary.csv", index=False)
    #     print("[INFO] Model comparison summary saved to output/TomSplit/model_comparison_summary.csv")
    
    # if all_per_class_results:
    #     per_class_master_df = pd.concat(all_per_class_results, ignore_index=True)
    #     per_class_master_df.to_csv("output/TomSplit/all_per_class_metrics_consolidated.csv", index=False)
    #     print("[INFO] Per-class results consolidated to output/TomSplit/all_per_class_metrics_consolidated.csv")


class Evaluation:
    def __init__(self, model_path, data_path, output_path, model_architecture, metrics_path=None, per_class_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.model_architecture = model_architecture
        self.metrics_path = metrics_path or output_path.replace('.csv', '_metrics.csv')
        self.per_class_path = per_class_path or output_path.replace('.csv', '_per_class_metrics.csv')

        # Class mapping configuration
        self.desired_order = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        if model_path.endswith("basemodel.keras"):
            self.train_order = ['P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
        else:
            self.train_order = self.desired_order
        self.train_to_desired = {self.train_order.index(cls): self.desired_order.index(cls) for cls in self.train_order}

    def remap_label(self, label):
        return self.train_to_desired.get(label, label)

    def load(self):
        print(f"[INFO] Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"[INFO] Loading dataset from {self.data_path}")
        self.data = PoseDataset(self.data_path)

        self.data.load_csv_data()
        reshape_flag = True if "cnn" in self.model_architecture.lower() or "attention" in self.model_architecture.lower() else False
        self.data.split_dataset_manual(test_size=0.2, reshape=reshape_flag, random_state=42)
        self.X_test = self.data.x_test
        if len(self.data.y_test.shape) > 1 and self.data.y_test.shape[1] > 1:
            self.y_test_sparse = np.argmax(self.data.y_test, axis=1)
        else:
            self.y_test_sparse = self.data.y_test

    def predict(self):
        print("[INFO] Running predictions on test set...")
        self.y_pred_probs = self.model.predict(self.X_test)
        self.y_pred_classes = self.y_pred_probs.argmax(axis=1)
        self.confidences = self.y_pred_probs.max(axis=1)
        
        # Apply remapping to predictions
        self.y_pred_remapped = [self.remap_label(int(y)) for y in self.y_pred_classes]

    def evaluate(self):
        # Overall metrics
        correct = sum(int(self.y_test_sparse[i]) == self.y_pred_remapped[i] for i in range(len(self.X_test)))
        self.acc = correct / len(self.X_test)
        self.precision = precision_score(self.y_test_sparse, self.y_pred_remapped, average='macro', zero_division=0)
        self.recall = recall_score(self.y_test_sparse, self.y_pred_remapped, average='macro', zero_division=0)
        
        print(f"[RESULT] Test set accuracy: {self.acc:.4f}")
        print(f"[RESULT] Test set precision: {self.precision:.4f}")
        print(f"[RESULT] Test set recall: {self.recall:.4f}")

    def calculate_per_class_metrics(self):
        """Calculate detailed per-class precision, recall, F1-score"""
        y_true = np.array(self.y_test_sparse)
        y_pred = np.array(self.y_pred_remapped)
        
        # Get unique classes (0-9 for P1-P10)
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )
        
        # Create detailed per-class results
        per_class_results = []
        for i, class_id in enumerate(classes):
            # Calculate confusion matrix components for this class
            tp = np.sum((y_pred == class_id) & (y_true == class_id))
            fp = np.sum((y_pred == class_id) & (y_true != class_id))
            fn = np.sum((y_pred != class_id) & (y_true == class_id))
            tn = np.sum((y_pred != class_id) & (y_true != class_id))
            
            per_class_results.append({
                'class_id': class_id,
                'class_name': self.desired_order[class_id] if class_id < len(self.desired_order) else f'Class_{class_id}',
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i],
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            })
        
        self.per_class_df = pd.DataFrame(per_class_results)
        self.per_class_df['accuracy'] = self.per_class_df['true_positives'] / (
            self.per_class_df['true_positives'] + self.per_class_df['false_negatives']
        )
        self.per_class_df['accuracy'] = self.per_class_df['accuracy'].fillna(0.0)

        # Calculate and verify macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        print(f"\n[VERIFICATION] Calculated macro precision: {macro_precision:.4f} (vs sklearn: {self.precision:.4f})")
        print(f"[VERIFICATION] Calculated macro recall: {macro_recall:.4f} (vs sklearn: {self.recall:.4f})")
        
        # Print per-class summary
        print("\nPer-Class Performance Summary:")
        print(self.per_class_df[['class_name', 'precision', 'recall', 'f1_score', 'support']].round(4).to_string(index=False))
        
        return self.per_class_df

    def save_metrics(self):
        """Save overall metrics"""
        metrics_df = pd.DataFrame([
            {
                "accuracy": self.acc,
                "precision": self.precision,
                "recall": self.recall
            }
        ])
        metrics_df.to_csv(self.metrics_path, index=False)
        print(f"[INFO] Overall metrics saved to: {self.metrics_path}")

    def save_per_class_metrics(self):
        """Save detailed per-class metrics"""
        self.per_class_df.to_csv(self.per_class_path, index=False)
        print(f"[INFO] Per-class metrics saved to: {self.per_class_path}")

    def save_results(self):
        """Save detailed predictions with confidence scores"""
        results = []
        for i in range(len(self.X_test)):
            results.append({
                "sample_index": i,
                "true_label": int(self.y_test_sparse[i]),
                "pred_label": self.y_pred_remapped[i],
                "confidence": float(self.confidences[i]),
                "features": ' '.join(map(str, self.X_test[i].flatten()))
            })
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_path, index=False)
        print(f"[INFO] Detailed predictions saved to: {self.output_path}")

    def run(self):
        self.load()
        self.predict()
        self.evaluate()
        self.calculate_per_class_metrics()
        self.save_metrics()
        self.save_per_class_metrics()
        self.save_results()


if __name__ == "__main__":
    main()