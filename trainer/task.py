"""A simple main file to showcase the template."""

import argparse
import logging.config
import os
import tensorflow as tf
from .model import build_model


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, activations
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from google.cloud import storage

"""
This module is an example for a single Python application with some
top level functions. The tests directory includes some unitary tests
for these functions.

This is one of two main files samples included in this
template. Please feel free to remove this, or the other
(sklearn_main.py), or adapt as you need.
"""

LOGGER = logging.getLogger()

def _list_files_by_prefix(bucket_name,prefix):

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=None
    )
    names = [blob.name for blob in blobs]
    return names


def _download_file(bucket_name, remote_name, dest_name):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(remote_name)
    blob.download_to_filename(dest_name)

    print(
        "Blob {} downloaded to {}.".format(
            remote_name, dest_name
        )
    )


def download_prepare_data(bucket_name, prefix):
    """Download and prepare the data for training

    Args:
      bucket_name: Name of the bucket
      prefix: prefix to the bathc
    """
    names = _list_files_by_prefix(bucket_name, prefix)

    # data/cat/cat.1.jpg


    for name in names:
        fn = name.split('/')[-1]
        if fn.endswith('jpg'):
            label = fn.split('.')[0]
            dest_dir = 'data/%s/' % label
            dest_name = 'data/%s/%s' % (label, fn)
            # Check that dest dir name
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            _download_file(bucket_name, name, dest_name)
        else:
            print("non jpg found")


def train_and_evaluate(bucket_name, prefix, download, img_size, batch_size,n_imgs, epochs):
    """build a dataimg generator"""

    #TODO: Read the Img and put it with the same size
    #model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),epochs=epochs,
    #                    validation_data=(x_test, y_test),
    #                    workers=4)

    #añadimos una condicion por so no las queremos descargar
    if download == True:
        download_prepare_data(bucket_name, prefix)
    else:
        print("Imgs already downloaded can be used")

    img_datagen = ImageDataGenerator(rescale=1.0 / 255)

    img_generator = img_datagen.flow_from_directory(
        'data',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    steps_per_epoch=int(n_imgs / batch_size)

    model = build_model(img_size)
    model.compile(optimizer=optimizers.Adam(), loss = losses.binary_crossentropy())
    model.fit_generator(img_generator, epochs=epochs, steps_per_epoch = steps_per_epoch)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument('--download', action = 'store_true') # si no ponemos nada por defecto es false siginifica que ya trnemos las img descargadas
    parser.add_argument("--epochs", required=True)
    args = parser.parse_args()

    bucket_name = args.bucket_name
    prefix = args.prefix
    epochs = args.epochs
    download = args.download
    
    train_and_evaluate(bucket_name, prefix, download, 128, 1, epochs)
