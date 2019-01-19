import tensorflow as tf
import numpy as np
from gcn import GeneralizedConvolutionalNetwork as gcn


def main():
    the_secret_sauce = gcn()

    random_seed = 123
    np.random.seed(random_seed)

    # Initialize our tensorflow graph environment
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(random_seed)

        the_secret_sauce.standardize_data()
        # Import our neural netsork into the tensorflow graph session
        the_secret_sauce.make_convolutional_neural_network()

        # Save our model to export it
        saver = tf.train.Saver()

    # Make a tensorflow session with our constructed graph
    with tf.Session(graph=g) as session:
        the_secret_sauce.train(session,
                               training_set=(
                                   the_secret_sauce._X_train,
                                   the_secret_sauce._y_train),
                               validation_set=(
                                   the_secret_sauce._X_valid,
                                   the_secret_sauce._y_valid),
                               initialize=True,
                               random_seed=123)
        the_secret_sauce.save_model(saver, session, epoch=20)


if __name__ == '__main__':
    main()
