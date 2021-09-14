# Import required packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense


class VGGNet:
    @staticmethod
    def build(hp):
        # Initialize model and input shape
        model = Sequential()
        channelDimension = -1

        # Block1: CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(180, 180, 3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block2: CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block3: CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('dropout1', 0.3, 0.5, step=0.1, default=0.5)))

        # First set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(hp.Int("dense_units", min_value=256, max_value=768, step=256)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout2', 0.3, 0.5, step=0.1, default=0.5)))

        # Softmax classifier
        model.add(Dense(7))
        model.add(Activation("softmax"))

        # initialize the learning rate choices and optimizer
        lr = hp.Choice("learning_rate",
                       values=[1e-1, 1e-2, 1e-3])
        opt = Adam(learning_rate=lr)
        # compile the model
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        # Return the model
        return model
