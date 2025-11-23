"""
Benchmarking with GroupKFold cross-validation using PoseDataset and DeepLearningModel.
This script splits the data by a fixed test group and uses GroupKFold for cross-validation on the remaining data.

python -m MainScripts.Benchmarking_GroupKF
"""

import os
import numpy as np
import pandas as pd
from Classes import PoseDataset, DeepLearningModel, ModelFactory
from sklearn.model_selection import GroupKFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Config
path_csv = "dtl_data_groupbynames.csv"
test_run_name = "DTL_groupKF"
folder = f"output/{test_run_name}"
os.makedirs(folder, exist_ok=True)

all_models = [
    ("MLP_Basic", ModelFactory.mlp_basic, True),
]

all_y_true = []
all_y_pred = []


for model_name, model_chosen, flatten in all_models:
    path_to_save_model = f"{folder}/{model_name}.h5"
    path_to_save_diagrams = f"{folder}/{model_name}"

    # Load and split dataset by group
    data = PoseDataset(path_csv)
    data.load_csv_data()
    # Use a fixed group for test set, e.g., group 5
    data.split_dataset_by_fixed_test_group(reshape=(not flatten), test_group_value=5, group_column='group', test_size=0.2)

    print('Train shape:', data.x_train.shape)
    print('Val shape:', data.x_val.shape)
    print('Test shape:', data.x_test.shape)
    print('Train groups:', np.unique(data.groups_train))
    print('Val groups:', np.unique(data.groups_val))
    print('Test groups:', np.unique(data.groups_test))

    # GroupKFold on train+val
    # X = np.concatenate([data.x_train, data.x_val])
    # y = np.concatenate([data.y_train, data.y_val])
    # groups = np.concatenate([data.groups_train, data.groups_val])

    # GroupKFold on EVERYTHING
    X = np.concatenate([data.x_train, data.x_val, data.x_test])
    y = np.concatenate([data.y_train, data.y_val, data.y_test])
    groups = np.concatenate([data.groups_train, data.groups_val, data.groups_test])

    n_splits=5
    gkf = GroupKFold(n_splits=n_splits)
    fold_accuracies = []
    fold_losses = []
    fold_f1_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_train_groups = []
    fold_val_groups = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"[GroupKF] Fold {fold+1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_groups = np.unique(groups[train_idx])
        val_groups = np.unique(groups[val_idx])
        print(f"    Train groups: {train_groups}")
        print(f"    Val groups:   {val_groups}")

        fold_train_groups.append("-".join(map(str, train_groups)))
        fold_val_groups.append("-".join(map(str, val_groups)))

        model = DeepLearningModel(
            input_shape = X_train.shape[1] if flatten else X_train.shape[1:],
            class_count = data.classCount,
            checkpoint_path = path_to_save_model,
            name = f"{model_name}_fold{fold+1}"
        )

        model.build_model(model_chosen)
        model.compile_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        history = model.model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=2
        )
        val_loss, val_acc = model.model.evaluate(X_val, y_val, verbose=0)
        print(f"    Fold {fold + 1} val_accuracy: {val_acc:.4f}")
        y_pred = model.model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_val_labels = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 and y_val.shape[1] > 1 else y_val

        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_val_labels, y_pred_labels, average='macro', zero_division=0)
        prec = precision_score(y_val_labels, y_pred_labels, average='macro', zero_division=0)
        rec = recall_score(y_val_labels, y_pred_labels, average='macro', zero_division=0)
        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)
        fold_f1_scores.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        print(f"[GroupKF] Fold {fold+1} - acc: {val_acc:.4f}, f1: {f1:.4f}, prec: {prec:.4f}, rec: {rec:.4f}")

        # Confusion matrix for this fold
        cm = confusion_matrix(y_val_labels, y_pred_labels)

        # Save as an image
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.allClasses, yticklabels=data.allClasses)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Fold {fold+1}, Val Group {fold_val_groups[-1]}")
        plt.title(f"Confusion Matrix - All Folds\nAcc={val_acc:.3f} | F1={f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f}")

        plt.tight_layout()
        plt.savefig(f"{path_to_save_diagrams}_confusion_fold{fold+1}_val{fold_val_groups[-1]}.png")
        plt.close()

        all_y_true.extend(y_val_labels)
        all_y_pred.extend(y_pred_labels)

        import tensorflow as tf
        tf.keras.backend.clear_session()

    # Save results
    cv_df = pd.DataFrame({
        'fold': range(1, len(fold_accuracies) + 1),
        'accuracy': fold_accuracies,
        'loss': fold_losses,
        'f1_score': fold_f1_scores,
        'precision': fold_precisions,
        'recall': fold_recalls,
        'train_groups': fold_train_groups,
        'val_groups': fold_val_groups
    })
    cv_df.to_csv(f"{path_to_save_diagrams}_groupkf_cv_results.csv", index=False)
    print(f"[INFO] GroupKF CV results saved to {path_to_save_diagrams}_groupkf_cv_results.csv")


    acc_total = accuracy_score(all_y_true, all_y_pred)
    f1_total = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    prec_total = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    rec_total = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)

    cm_total = confusion_matrix(all_y_true, all_y_pred)

    # Aggregate metrics (from all_y_true / all_y_pred)
    acc_total = accuracy_score(all_y_true, all_y_pred)
    f1_total = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    prec_total = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    rec_total = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)

    # Average metrics (from per-fold lists)
    acc_avg = np.mean(fold_accuracies)
    f1_avg = np.mean(fold_f1_scores)
    prec_avg = np.mean(fold_precisions)
    rec_avg = np.mean(fold_recalls)

    # Confusion matrix
    cm_total = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues",
                xticklabels=data.allClasses, yticklabels=data.allClasses)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add both sets of scores in the title
    plt.title(
        "Confusion Matrix - All Folds\n"
        f"Aggregate → Acc={acc_total:.3f} | F1={f1_total:.3f} | Prec={prec_total:.3f} | Rec={rec_total:.3f}\n"
        f"Average   → Acc={acc_avg:.3f} | F1={f1_avg:.3f} | Prec={prec_avg:.3f} | Rec={rec_avg:.3f}"
    )

    plt.tight_layout()
    plt.savefig(f"{path_to_save_diagrams}_confusion_all_folds.png")
    plt.close()

    # Optionally, evaluate on the fixed test set
    # print("[INFO] Evaluating on fixed test set...")
    # model = DeepLearningModel(
    #     input_shape = data.x_train.shape[1] if flatten else data.x_train.shape[1:],
    #     class_count = data.classCount,
    #     checkpoint_path = path_to_save_model,
    #     name = model_name
    # )
    # model.build_model(model_chosen)
    # model.compile_model()
    # model.model.fit(
    #     np.concatenate([data.x_train, data.x_val]),
    #     np.concatenate([data.y_train, data.y_val]),
    #     epochs=200,
    #     batch_size=16,
    #     verbose=2
    # )
    # test_loss, test_acc = model.model.evaluate(data.x_test, data.y_test, verbose=0)
    # print(f"[INFO] Test set accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
