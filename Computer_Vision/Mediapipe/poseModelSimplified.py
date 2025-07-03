import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import os 
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from Classes import PoseDataset, DeepLearningModel, ModelFactory
import argparse
from keras import layers, Sequential, regularizers

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", type=str, required=True,
#                 help="path to csv Data")
# ap.add_argument("-o", "--save", type=str, required=True,
#                 help="path to save .h5 model, eg: dir/model.h5")
# args = vars(ap.parse_args())
# path_csv = args["dataset"]
# path_to_save = args["save"]


all_models = [
    ("MLP_Basic", ModelFactory.mlp_basic, True),
    ("MLP_Deep", ModelFactory.mlp_deep, True),
    ("MLP_Dropout", ModelFactory.mlp_with_dropout, True),
    ("MLP_Attention", ModelFactory.mlp_attention, False),
    ("CNN_Basic", ModelFactory.cnn_basic, False),
    ("CNN_Attention", ModelFactory.cnn_attention, False),
    ("CNN_2D", ModelFactory.cnn_2d, False),
    ("CNN_3_block", ModelFactory.cnn_3_block, False) # architecture from paper
]

results = {}
path_csv = "dataset3.csv"
# path_csv = "test_set.csv"
test_run_name = "D3_"
folder = f"output/{test_run_name}"
os.makedirs(folder, exist_ok=True)


for m in all_models:
    model_name, model_chosen, flatten = m[0], m[1], m[2]
    path_to_save_model = f"{folder}/{model_name}.h5"
    path_to_save_diagrams = f"{folder}/{model_name}"

    # initialise dataset
    data = PoseDataset(path_csv)
    data.load_csv_data()
    # default: test_size=0.2, random_state=0
    # data.split_dataset(test_size=0.3, reshape=(not flatten), random_state=42)
    data.split_dataset_manual(reshape=(not flatten))
    

    # initialise model
    model = DeepLearningModel(
        input_shape = data.x_train.shape[1] if flatten else data.x_train.shape[1:],
        class_count = data.classCount,
        checkpoint_path = path_to_save_model,
        name = model_name
    )

    # a function can be passed in to change the model architecture, otherwise it will use default model (from seniors)
    # signature: fn(input_shape, class_count)
    model.build_model(
        model_chosen
    )

    #todo: could possibly add customisation type of optimiser, loss, metrics
    model.compile_model()

    #todo: could possibly add customisation to automatically add type of callbacks
    model.add_callbacks()

    # other params: epoch (default 200), batch_size (default 16)
    model.train(data, epochs=500)

    model.plot_training_metrics(path_to_save_diagrams)

    model.plot_confusion_matrix(data, path_to_save_diagrams, "val")
    model.plot_confusion_matrix(data, path_to_save_diagrams, "test")

    results[model_name] = model.valResults + model.testResults

    columns = ["val_acc", "val_prec", "val_rec", "val_f1", "test_acc", "test_prec", "test_rec", "test_f1"]
    print(results)
    df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
    df_rounded = df.round(3)
    df_rounded.to_csv(f"{folder}/Results.csv")
    print(f"[INFO] Saved in {folder}/Results.csv")