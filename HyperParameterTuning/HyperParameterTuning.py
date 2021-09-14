# Import required packages
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from Vggnet import VGGNet
from load_dataset import load
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
from imutils import paths
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tuner", default="hyperband", type=str, choices=["hyperband", "random"],
                help="type of hyperparameter tuner we'll be using")
args = vars(ap.parse_args())

# Grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images("/Users/advaitdixit/AppliedDLHPE/Images/simple_images"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
(data, labels) = load(width=180, height=180, imagePaths=imagePaths, verbose=100)
data = data.astype("float") / 255.0

# Split training-testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Initialize an early stopping callback to prevent the model from overfitting
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Check if we will be using the hyperband tuner
if args["tuner"] == "hyperband":
    # Instantiate the hyperband tuner object
    print("[INFO] instantiating a hyperband tuner object...")
    tuner = kt.Hyperband(
        VGGNet.build,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        seed=42,
        directory="output",
        project_name=args["tuner"])

# Check if we will be using the random search tuner
elif args["tuner"] == "random":
    # Instantiate the random search tuner object
    print("[INFO] instantiating a random search tuner object...")
    tuner = kt.RandomSearch(
        VGGNet.build,
        objective="val_accuracy",
        max_trials=10,
        seed=42,
        directory="output",
        project_name=args["tuner"])

# Perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
    x=trainX, y=trainY,
    validation_data=(testX, testY),
    batch_size=32,
    callbacks=[es],
    epochs=10
)

# Grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of units in dense layer: {}".format(bestHP.get("dense_units")))
print("[INFO] optimal learning rate: {:.4f}".format(bestHP.get("learning_rate")))
print("[INFO] optimal dropout rate 1: {:.4f}".format(bestHP.get("dropout1")))
print("[INFO] optimal dropout rate 2: {:.4f}".format(bestHP.get("dropout2")))

# Build the best model and train it
print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), batch_size=32, epochs=20, callbacks=[es], verbose=1)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))