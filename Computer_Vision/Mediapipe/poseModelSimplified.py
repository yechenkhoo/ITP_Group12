from Classes import PoseDataset, DeepLearningModel
import argparse
from keras import layers, Sequential, regularizers

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to csv Data")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save .h5 model, eg: dir/model.h5")
args = vars(ap.parse_args())
path_csv = args["dataset"]
path_to_save = args["save"]

# initialise dataset
data = PoseDataset(path_csv)
data.load_csv_data()
# default: test_size=0.2, random_state=0
data.split_dataset(test_size=0.3)

# initialise model
model = DeepLearningModel(
    input_shape = data.x_train.shape[1],
    class_count = data.classCount,
    checkpoint_path = path_to_save,
)

# a function can be passed in to change the model architecture, otherwise it will use default model (from seniors)
# signature: fn(input_shape, class_count)
model.build_model(
    # lambda inputShape, classCount: Sequential([
    #     layers.Dense(512, activation='relu', input_shape=[inputShape], 
    #                 kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(classCount, activation="softmax")
    # ])
)

#todo: could possibly add customisation type of optimiser, loss, metrics
model.compile_model()

#todo: could possibly add customisation to automatically add type of callbacks
model.add_callbacks()

# other params: epoch (default 200), batch_size (default 16)
model.train(data)

model.plot_training_metrics(path_to_save)

model.plot_confusion_matrix(data, path_to_save, "val")
model.plot_confusion_matrix(data, path_to_save, "test")
