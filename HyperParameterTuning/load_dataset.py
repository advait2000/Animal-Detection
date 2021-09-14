# import the necessary packages
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load(width, height, imagePaths, verbose=100):
    # initialize the list of features and labels
    data = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label
        image = load_img(imagePath, target_size=(width, height))
        image = img_to_array(image, data_format=None)
        label = imagePath.split(os.path.sep)[-2]

        # treat our processed image as a "feature vector"
        # by updating the data list followed by the labels
        data.append(image)
        labels.append(label)

        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    # return a tuple of the data and labels
    return np.array(data), np.array(labels)
