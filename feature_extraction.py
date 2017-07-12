import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
import numpy as np
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from keras.models import Sequential
from keras.utils import to_categorical

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('epochs', 50, "Number of epochs")
flags.DEFINE_string('batch_size', 256, "Batch size")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

def sequential_model():
    nb_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(nb_classes, activation='softmax'),
        ])

def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    nb_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

#    inputs = Input(shape=input_shape)
#
#    # a layer instance is callable on a tensor, and returns a tensor
#    flatten = Flatten()(inputs)
#    predictions = Dense(nb_classes, activation='softmax')(flatten)
#
#    # This creates a model that includes
#    # the Input layer and three Dense layers
#    model = Model(inputs=inputs, outputs=predictions)

    print(input_shape)
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(nb_classes),
        Activation('softmax'),
        ])

#    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    y_train_one_hot = to_categorical(y_train, num_classes=nb_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=nb_classes)
#    model.fit(X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)
    model.fit(X_train, y_train_one_hot, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val_one_hot), shuffle=True)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
