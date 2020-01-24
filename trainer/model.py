"""a model."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, activations

def build_model(img_size):
    """Return a neural network"""

    shape = (img_size, img_size, 3)

    m = models.Sequential()
    m.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(64, (3, 3), activation='relu'))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(1, activation='sigmoid'))

    return m
