import numpy as np
import pandas as pd
import random
import mediapipe as mp
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

epochs_count = 200
folder_path = "output/T_gnn/"

def main():
    pose_vector = pd.read_csv("dataset4.csv")
    image_paths = pose_vector.pop('Image_Path').to_list()  # store image paths
    y = pose_vector.pop('Pose_Class').to_numpy()
    y = [int(val.replace('P', '')) - 1 for val in y]
    points = pose_vector.to_numpy()
    pose_vector = points.reshape((-1, 33, 4))
    print(f"Pose Vector Shape: {pose_vector.shape}")

    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS
    print(f"MediaPipe list of connections: {list(connections)}")

    connections_array = np.array(list(connections)).T
    print(f"Connections array: {connections_array} \n -> Shape {connections_array.shape}")
    edge_index = torch.tensor(connections_array, dtype=torch.long)  # convert edges to LongTensor
    
    # Prepare Dataset
    # x = pose_vector, y is label array (0 to 9), edge_index is from body connections
    train_dataset, val_dataset, test_dataset = split_gnn_dataset_manual(pose_vector, y, edge_index, image_paths=image_paths)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialise model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(input_dim=4, hidden_dim=64, output_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs_count+1):
        print(epoch)
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn=loss_fn)

        # Store values
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


    # Plot training curve
    epochs = range(1, epochs_count+1)
    plot_loss_and_accuracy(epochs, train_losses, val_losses, val_accuracies)

    # Plot confusion matrices
    # Validation matrix
    plot_confusion_matrix_gnn(
        model=model,
        data_loader=val_loader,
        device=device,
        path_to_save=folder_path,
        all_classes=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
        dataset_name="val"
    )

    # Test matrix
    plot_confusion_matrix_gnn(
        model=model,
        data_loader=test_loader,
        device=device,
        path_to_save=folder_path,
        all_classes=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
        dataset_name="test"
    )



class PoseDataset(Dataset):
    def __init__(self, pose_data, labels, edge_index, image_paths=None):
        super().__init__()
        self.pose_data = pose_data  # shape: [num_samples, 33, 4] - 33 keypoints with 4d features
        self.labels = labels        # shape: [num_samples] 
        self.edge_index = edge_index  # shape: [2, num_edges] - edges that define graph structure
        self.image_paths = image_paths if image_paths is not None else [None] * len(pose_data)

    def __len__(self):
        return len(self.pose_data) # total num of samples

    def __getitem__(self, idx):
        x = torch.tensor(self.pose_data[idx], dtype=torch.float)  # one sample's features [33, 4]
        y = torch.tensor(self.labels[idx], dtype=torch.long) # label
        img_path = self.image_paths[idx]
        return Data(x=x, edge_index=self.edge_index, y=y, img_path=img_path)


class GNNModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=10):  # output_dim = num classes
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim) # first layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim) # second layer
        self.fc = nn.Linear(hidden_dim, output_dim) # final classifier

    def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch  # x: [N, 4]

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

            x = global_mean_pool(x, batch)  # shape: [batch_size, hidden_dim]
            x = self.fc(x)
            return x


# class GNNModel(nn.Module):
#     def __init__(self, input_dim=4, hidden_dim=64, output_dim=10, dropout=0.5):
#         super(GNNModel, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.bn1 = BatchNorm1d(hidden_dim)
        
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = BatchNorm1d(hidden_dim)

#         self.fc1 = Linear(hidden_dim, hidden_dim // 2)
#         self.fc2 = Linear(hidden_dim // 2, output_dim)
#         self.dropout = Dropout(p=dropout)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # First GCN layer
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)

#         # Second GCN layer
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)

#         # Global Pooling
#         x = global_mean_pool(x, batch)

#         # Fully connected MLP head
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

def split_gnn_dataset(pose_vector, labels, edge_index, image_paths=None, test_size=0.2, random_state=0):
    points_trainval, points_test, y_trainval, y_test, paths_trainval, paths_test = train_test_split(
        pose_vector, labels, image_paths, test_size=test_size, random_state=random_state, stratify=labels
    )
    points_train, points_val, y_train, y_val, paths_train, paths_val = train_test_split(
        points_trainval, y_trainval, paths_trainval, test_size=test_size, random_state=random_state, stratify=y_trainval
    )

    def to_dataset(points, y, paths):
        return PoseDataset(points, y, edge_index, paths)

    return (
        to_dataset(points_train, y_train, paths_train),
        to_dataset(points_val, y_val, paths_val),
        to_dataset(points_test, y_test, paths_test)
    )

def split_gnn_dataset_manual(pose_vector, labels, edge_index, image_paths=None, val_size=0.2, random_state=0):
    """
    Splits the dataset manually:
    - Test set: entries where 'image_paths' contains 'tom'
    - Train/Val: all others
    - Validation set is a fraction (val_size) of train set
    """
    image_paths = np.array(image_paths)
    pose_vector = np.array(pose_vector)
    labels = np.array(labels)

    # Identify indices for Charlie and non-Charlie samples
    test_indices = [i for i, p in enumerate(image_paths) if "tom" in p]
    trainval_indices = [i for i in range(len(image_paths)) if i not in test_indices]

    # Split into Charlie test set and train/val
    points_test = pose_vector[test_indices]
    y_test = labels[test_indices]
    paths_test = image_paths[test_indices]

    points_trainval = pose_vector[trainval_indices]
    y_trainval = labels[trainval_indices]
    paths_trainval = image_paths[trainval_indices]

    # Split train/val using sklearn
    points_train, points_val, y_train, y_val, paths_train, paths_val = train_test_split(
        points_trainval, y_trainval, paths_trainval, test_size=val_size,
        random_state=random_state, stratify=y_trainval
    )

    def to_dataset(points, y, paths):
        return PoseDataset(points, y, edge_index, paths)

    return (
        to_dataset(points_train, y_train, paths_train),
        to_dataset(points_val, y_val, paths_val),
        to_dataset(points_test, y_test, paths_test)
    )

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    return total_loss / total, correct / total

def plot_loss_and_accuracy(epochs, train_losses, val_losses, val_accuracies):
    # Plot Loss over epochs and save as figure
    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f"{folder_path}loss_accuracy_over_epochs_{epochs_count}.png", bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved loss and accuracy plot at {folder_path}loss_accuracy_over_epochs_{epochs_count}.png")


def plot_confusion_matrix_gnn(
    model, data_loader, device, path_to_save, all_classes, dataset_name="val", save_csv=True
):
    model.eval()
    y_true = []
    y_pred_classes = []
    all_records = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data)  # raw scores
            probs = F.softmax(logits, dim=1).cpu().numpy()  # shape: [batch_size, num_classes]

            preds = logits.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            img_paths = getattr(data, 'img_path', [None] * len(labels))
            x = data.x.cpu()
            batch = data.batch.cpu()

            for i in range(len(labels)):
                y_true.append(labels[i])
                y_pred_classes.append(preds[i])

                node_mask = (batch == i)
                pose = x[node_mask].numpy()  # shape: [33, 4]
                if pose.shape[0] != 33:
                    print(f"[WARN] Sample {i} has {pose.shape[0]} landmarks, expected 33 — skipping")
                    continue
                landmarks_flat = pose.flatten().tolist()

                record = {
                    "img_path": img_paths[i],
                    "true_label": labels[i],
                    "pred_label": preds[i],
                    "confidence": probs[i][preds[i]],
                    "landmarks": ' '.join(map(str, landmarks_flat))
                }
                all_records.append(record)

    # Save as CSV
    if save_csv:
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(os.path.dirname(path_to_save), f"{dataset_name}_predictions{epochs_count}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved prediction CSV with confidence at {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    acc = accuracy_score(y_true, y_pred_classes)
    prec = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)

    # Plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{dataset_name.capitalize()} Confusion Matrix\n"
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    fig_path = f"{path_to_save}/{dataset_name}_confusion_matrix_{epochs_count}.png"
    if os.path.exists(fig_path):
        os.remove(fig_path)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved confusion matrix for {dataset_name} set at {fig_path}")


main()