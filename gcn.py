import os
import struct
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image


class GeneralizedConvolutionalNetwork(object):
    def __init__(self):
        self._labels = None
        self._images = None

        self._X_train = None
        self._y_train = None

        self._X_test = None
        self._y_test = None

        self._X_valid = None
        self._y_valid = None

        self.width = 96
        self.height = 96

    def batch_shape_image_data(
            self, images: list, inplace: bool = False)->list:
        for image in images:
            # Get image path for remaking it into a PNG
            filename, _ = os.path.splitext(image)

            # First, load the image
            img = Image.open(image)
            img.save(filename + '.png')
            the_png = Image.open(filename + '.png')
            new_img = the_png.resize((self.width, self.height))

            # Save our new 96x96 image
            new_img.save(filename + '.png')

    def load_image_data(self, names: list, annotations: list)->tuple:
        name_annotations = dict.fromkeys(names)

        for annotation in annotations:
            if annotation[0] in name_annotations:
                name_annotations[annotation[0]] = annotation[1]

        image_data_frame = pd.DataFrame(
                name_annotations.items(), columns=['image_name', 'annotation'])

        return image_data_frame

    def standardize_data(self, frame: pd.DataFrame)->None:
        sc = StandardScaler()
        X = frame['image_name']
        X = sc.transform(X)

        return X

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
            padding_mode: str = 'SAME',
            strides: tuple = (1, 1, 1, 1)):
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

            conv = tf.nn.bias_add(conv, biases,
                                  name='net_pre_activation')

            # Apply our activation function to our network
            conv = tf.nn.relu(conv, name='activation')

            return conv

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
        reduces the dimensionality of the data at each convolution to
        facilitate greater computational performance, and less needed
        system resources. We then tie to a final fully connected multilayer
        perceptron to perform our final prediction tasks

        Parameters
        ----------
        learning_rate: float - The learning rate for how big of jumps we want
                               to make when running our optimizer
        '''
        # Image size is 9216 because 96x96 image sizes
        tf_x = tf.placeholder(tf.float32, shape=[None, 9216], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        # Reshape our x to a 4-dimensional tensor:
        # shape = [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                                name='tf_x_reshaped')
        # One hot encode our class labels
        tf_y_onehot = tf.one_hot(
                indices=tf_y, depth=10, dtype=tf.float32, name='tf_y_onehot')

        # First layer: Convolution one
        layer_1 = self.convolutional_layer(tf_x_image, name='conv_1',
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
        layer_2 = self.convolutional_layer(layer_1_pool, name='conv_2',
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
            # Use a softmax axcitvation function on
            # our output layer to get our probs
            'probabilities': tf.nn.softmax(layer_4, name='probabilities'),
            'labels': tf.cast(tf.argmax(
                layer_4, axis=1), tf.int32, name='labels')
        }

        # Loss function and optimization and add it to our graph
        cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=layer_4, labels=tf_y_onehot),
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

        print('Accuracy: {}'.format(accuracy))

    def save_model(self, saver, sess, epoch, path='./model'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model{}'.format(path))
        saver.save(sess, os.path.join(
            path, 'cnn-model.ckpt'), glboal_step=epoch)

    def load(self, saver, sess, path, epoch):
        print('Loading model from: {}'.format(path))
        saver.restore(sess, os.path.join(
            path, 'cnn-model.ckpt-{}'.format(epoch)))

    def train(self, sess, training_set, validation_set=None,
              initialize=True, epochs=20, shuffle=True,
              dropout=0.5, random_seed=None):
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        training_loss = []

        if initialize:
            sess.run(tf.global_variables_initializer())

        np.random.seed(random_seed)

        for epoch in range(1, epochs + 1):
            batch_gen = self.generate_batch(X_data, y_data, shuffle=shuffle)
            avg_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y,
                        'fc_keep_prob:0': dropout}
                loss, _ = sess.run(
                        ['cross_entropy_loss:0', 'train_op'],
                        feed_dict=feed)
                avg_loss += loss

            training_loss.append(avg_loss / (i + 1))
            print('Epoch {} Training Avg. Loss: {}'
                  .format(epoch, avg_loss), end=' ')

            if validation_set is not None:
                feed = {'tf_x:0': validation_set[0],
                        'tf_y:0': validation_set[1],
                        'fc_keep_prob:0': 1.0}
                valid_acc = sess.run('accuracy:0', feed_dict=feed)
                print(' Validation Acc: {}'.format(valid_acc))
            else:
                print()

    def predict(self, sess, X_test, return_probability=False):
        feed = {'tf_x:0': X_test,
                'fc_keep_prob:0': 1.0}
        if return_probability:
            return sess.run('probabilities:0', feed_dict=feed)
        else:
            return sess.run('labels:0', feed_dict=feed)
