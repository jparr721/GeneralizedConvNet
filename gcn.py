import sys
import gzip
import shutil
import os
import struct
import tensorflow as tf
import numpy as np


class GeneralizedConvolutionalNetwork(object):
    def __init__(self):
        pass

    def load_image_data(self, path: str, kind: str = 'train')->tuple:
        pass

    def generate_batch(X: np.array, y: np.array, batch_size: int = 64,
                       shuffle: bool = False, random_seed: int = None):
        # Takes the length of y at dimension 1 to get the index range
        # of our class labels
        indexes = np.arange(y.shape[0])

        # If shuffle flag set, randomize our data
        if shuffle:
            random_number_generator = np.random.RandomState(random_seed)
            # Shuffle our indexes so they are always in range
            random_number_generator.shuffle(indexes)

            # Remap our X and y to fit the indexes
            X = X[indexes]
            y = y[indexes]

        # Generate nth batch
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i+batch_size, :], y[i:i+batch_size])

    def convolutional_layer(
            self,
            input_tensor,
            name: str,
            kernel_size: tuple,
            n_output_channels: int,
            padding_mode: str ='SAME',
            strides: tuple =(1, 1, 1, 1)):
        '''
        The convolutional layer constructs a tensorflow convolution
        to facilitate reduction of code duplication. This module can
        be used to generate more convolutions as needed

        Parameters
        ---------
        input_tensor - The input tensorflow tensor matrix (nd)
        name: str - The name of our convolutional layer for reference
                    in the tensorflow graph
        kernel_size: tuple - The dimensionality of the kernel we will use
                             (example: (3, 3) is a 3x3 dimension)
        n_output_channels: int - The number of outputs we want in our
                                  convolutional layers
        padding_modde: str - This is the type of padding we use in the
                             convolutional image processing layer.
                             Certain padding inhibits the original
                             image quality, so SAME is usually a good bet
                             when dealing with classification tasks, which
                             is why that is the default mode.
        '''

        # Lets us create a reusable layer in the
        # tensorflow graph without needing to wrap
        # everything in a session right away
        with tf.variable_scope(name):
            # get our n input channels
            # shape: [batch_size x width x height x input channels]
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1]

            weights_shape = (list(kernel_size) +
                             [n_input_channels, n_output_channels])

            ''' NOTE: We should change these names if we need to '''
            # Get weight from the tf graph or create with existing shape
            weights = tf.get_variable(name='_weights',
                                      shape=weights_shape)

            # Biases adjust our network representation along our activation
            # function graph, these can vary dependeing what outcome we want

            # These will have all zeros matching our
            # output channel number to begin with
            biases = tf.get_variable(name='_biases',
                                     initializer=tf.zeros(
                                         shape=[n_output_channels]))

            # Generate a 2-d convolution given 4d input
            conv = tf.nn.conv2d(input=input_tensor,
                                filter=weights,
                                strides=strides,
                                padding=padding_mode)

    def fully_connected_layer(
            self,
            input_tensor,
            name: str,
            n_output_units: int,
            activation_fn=None):
        '''
        The fully connected layer sits at the front of the convolutional
        neural network and maps the deeper layers to the output. This allows
        each convolutional and pooling layer to be mapped to every weight
        and output unit in the final layer.

        Parameters
        ----------
        input_tensor - The input tensorflow rank-n tensor
        name: str - The name of our convolutional layer for reference
                    in the tensorflow graph
        n_output_units: int - The number of outputs we are mapping to
        activation_function - The activation function we use on each
                              node in our perceptron
        '''
        with tf.variable_scope(name):

            input_shape = input_tensor.get_shape().as_list()[1:]
            n_input_units = np.prod(input_shape)
            if len(input_shape) > 1:
                input_tensor = tf.reshape(input_tensor,
                                          shape=(-1, n_input_units))

            weights_shape = [n_input_units, n_output_units]
            weights = tf.get_variable(name='_weights',
                                      shape=weights_shape)
            print('weights: {}'.format(weights))

            biases = tf.get_variable(name='_biases',
                                     initializer=tf.zeros(
                                        shape=[n_output_units]))
            print('biases: {}'.format(biases))

            layer = tf.matmul(input_tensor, weights)
            print('layer: {}'.format(layer))
            layer = tf.nn.bias_add(layer, biases, name='activation')
            print('layer: {}'.format(layer))

            if activation_fn is None:
                return layer

            layer = activation_fn(layer, name='activation')
            print('layer: {}'.format(layer))
            return layer

    def make_convolutional_neural_network(self, learning_rate: float = 1e-4):
        '''
        This function creates the full convolutional network by combining
        several convolutional layers with a pooling layer. The pooling layer
        reduces the dimensionality of the data at each convolution to facilitate
        greater computational performance, and less needed system resources. We then
        tie to a final fully connected multilayer perceptron to perform our final
        prediction tasks

        Parameters
        ----------
        learning_rate: float - The learning rate for how big of jumps we want to
                               make when running our optimizer
        '''
        # NOTE: The shape of this WILL CHANGE when we get the image dimens.
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        # Reshape our x to a 4-dimensional tensor:
        # shape = [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                                name='tf_x_reshaped')
        # One hot encode our class labels
        tf_y_onehot = tf.one_hot(
                indices=tf_y, depth=10, dtype=tf.float32, name='tf_y_onehot')

        # First layer: Convolution one
        layer_1 = self.conv_layer(tf_x_image, name='conv_1',
                                  kernel_size=(5, 5),
                                  padding_mode='VALID',
                                  n_output_channels=32)

        # Add our max pooling layer to minimize image size
        layer_1_pool = tf.nn.max_pool(layer_1,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')

        # Second Layer: Convolution two
        # We add the layer 1 pool into the second convolutional
        # layer to further optimize on more granular results
        layer_2 = self.conv_layer(layer_1_pool, name='conv_2',
                                  kernel_size=(5, 5),
                                  padding_mode='VALID',
                                  n_output_channels=64)

        # Add our max pooling for layer two
        layer_2_pool = tf.nn.max_pool(layer_2,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')

        # Third Layer: Fully connected layer
        layer_3 = self.fully_connected_layer(layer_2_pool, name='fc_3',
                                             n_output_units=1024,
                                             activation_fn=tf.nn.relu)

        # Compute our dropout to prevent overfit
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        layer_3_dropout = tf.nn.dropout(
                layer_3, keep_prob=keep_prob, name='dropout_layer')

        # Fouth Layer: Fully connected, linear activation
        # Computes with linear activation and dropout
        layer_4 = self.fully_connected_layer(layer_3_dropout, name='fc_4',
                                             n_output_units=10)

        # Make our predictions
        predictions = {
            # Use a softmax axcitvation function on our output layer to get our probs
            'probabilities': tf.nn.softmax(layer_4, name='probabilities'),
            'labels': tf.cast(tf.argmax(layer_4, axis=1), tf.in32, name='labels')
        }


        # Loss function and optimization and add it to our graph
        cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=h4, labels=tf_y_onehot),
                name='cross_entropy_loss')

        # Our optimizer function
        # For the Adam optimizer https://arxiv.org/pdf/1412.6980.pdf
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

         # Computing the prediction accuracy
        correct_predictions = tf.equal(
            predictions['labels'],
            tf_y, name='correct_preds')

        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy')