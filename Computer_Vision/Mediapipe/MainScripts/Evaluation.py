"""
Run this using:
python -m MainScripts.Evaluation
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from Classes.PoseDataset import PoseDataset
from sklearn.metrics import precision_score, recall_score


def main():
    # CONFIG VARIABLES
    # MODEL_PATH = "models/basemodel.keras"
    naming_convention="output/TomSplit/hyperparameter_tuning_kerastuner_"
    tuning_types=["bayesian", "random"]
    layer_tunings=["_layertuning", ""]
    model_architectures=["cnn", "mlp"]
    # MODEL_PATH = "output/TomSplit/hyperparameter_tuning_kerastuner_bayesian_tomsplit/mlp/best_model.keras"
    # OUTPUT_PATH = "output/TomSplit/HTKBT_MLP_confidence_eval.csv"
    DATA_PATH = "dataset4.csv"

    all_results = []
    for tuning_type in tuning_types:
        for layer_tuning in layer_tunings:
            for model_architecture in model_architectures:
                model_path = f"{naming_convention}{tuning_type}{layer_tuning}_tomsplit/{model_architecture}/best_model.keras"
                output_path = f"{naming_convention}{tuning_type}{layer_tuning}_tomsplit/{model_architecture}/eval.csv"
                metrics_path = output_path.replace('.csv', '_metrics.csv')
                evaluator = Evaluation(
                    model_path=model_path,
                    data_path=DATA_PATH,
                    output_path=output_path,
                    metrics_path=metrics_path,
                    model_architecture=model_architecture
                )
                evaluator.run()
                df = pd.read_csv(metrics_path)
                df['tuning_type'] = tuning_type
                df['layer_tuning'] = layer_tuning if layer_tuning else 'none'
                df['model_architecture'] = model_architecture
                all_results.append(df)
    # Concatenate and save
    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_df.to_csv("output/TomSplit/all_evaluations_consolidated.csv", index=False)
        print("[INFO] All results consolidated to output/all_evaluations_consolidated.csv")


class Evaluation:
    def __init__(self, model_path, data_path, output_path, model_architecture, metrics_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.model_architecture = model_architecture
        self.metrics_path = metrics_path or output_path.replace('.csv', '_metrics.csv')

        # PREVIOUS TEAM USED P1, P10, P2, P3, P4, P5, P6 ..., P9.
        # Currently we use P1 to P10 in order, hence if we use basemodel (old model), we would need to remap the data
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
        reshape_flag = True if self.model_architecture != "mlp" else False
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

    def evaluate(self):
        correct = sum(int(self.y_test_sparse[i]) == self.remap_label(int(self.y_pred_classes[i])) for i in range(len(self.X_test)))
        self.acc = correct / len(self.X_test)
        self.precision = precision_score(self.y_test_sparse, [self.remap_label(int(y)) for y in self.y_pred_classes], average='macro', zero_division=0)
        self.recall = recall_score(self.y_test_sparse, [self.remap_label(int(y)) for y in self.y_pred_classes], average='macro', zero_division=0)
        print(f"[RESULT] Test set accuracy: {self.acc:.4f}")
        print(f"[RESULT] Test set precision: {self.precision:.4f}")
        print(f"[RESULT] Test set recall: {self.recall:.4f}")

    def save_metrics(self):
        metrics_df = pd.DataFrame([
            {
                "accuracy": self.acc,
                "precision": self.precision,
                "recall": self.recall
            }
        ])
        metrics_df.to_csv(self.metrics_path, index=False)
        print(f"[INFO] Metrics saved to: {self.metrics_path}")

    def save_results(self):
        results = []
        for i in range(len(self.X_test)):
            results.append({
                "sample_index": i,
                "true_label": int(self.y_test_sparse[i]),
                "pred_label": self.remap_label(int(self.y_pred_classes[i])),
                "confidence": float(self.confidences[i]),
                "features": ' '.join(map(str, self.X_test[i].flatten()))
            })
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_path, index=False)
        print(f"[INFO] Confidence evaluation saved to: {self.output_path}")

    def run(self):
        self.load()
        self.predict()
        self.evaluate()
        self.save_metrics()
        self.save_results()

if __name__ == "__main__":
    main()
