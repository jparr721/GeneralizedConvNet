# GeneralizedConvNet
This is a (mostly) generalized convolutional neural network architecture for image classificaton. The network focuses on high granularity for
highly accurate prediction

## Using this
Import the module into your code:
```python
from gcn import GeneralizedConvolutionalnetwork as GCN
import numpy as np
import tensorflow as tf
```

Then simply run the module training function inside of your main method (or wherever you want your data to go)
```python
def main():
  g = tf.Graph()
  with g.as_default():
      tf.set_random_seed(random_seed)
      ## build the graph
      build_cnn()

      ## saver:
      saver = tf.train.Saver()

  with tf.Session(graph=g) as sess:
      train(sess,
            training_set=(X_train_centered, y_train),
            validation_set=(X_valid_centered, y_valid),
            initialize=True,
            random_seed=123)
      save(saver, sess, epoch=20)
      preds = predict(sess, X_test_centered,
                      return_proba=False)

      print('Test Accuracy: %.3f%%' % (100*
                  np.sum(preds == y_test)/len(y_test)))
```

This code might change depending on what you need, so don't use it as a "this will definitely work as-is" and use it rather as an example to go off of.
