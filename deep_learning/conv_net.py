"""Convolutional Neural Net using skflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import tensorflow as tf

layers = tf.contrib.layers
learn = tf.contrib.learn


def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode,
               image_dim, layer1_size, layer2_size,
               dense_layer_size,
               num_classes=2, num_colors=1,
               kernel_size=[5,5]):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, num_classes)
  # and with a on-value of 1 for each one-hot vector of length num_classes.
  target = tf.one_hot(tf.cast(target, tf.int32), num_classes, 1, 0)

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, image_dim[0], image_dim[1], num_colors])

  # First conv layer will compute layer1_size features for each
  # kernel_size patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution2d(
        feature, layer1_size, kernel_size=kernel_size, activation_fn=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)

  # Second conv layer will compute layer2_size features for each
  # kernel_size patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution2d(
        h_pool1, layer2_size, kernel_size=kernel_size, activation_fn=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * layer2_size])

  # Densely connected layer with dense_layer_size neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, dense_layer_size, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, num_classes, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op


def main(unused_arg):
  ### Download and load MNIST dataset.
  mnist = learn.datasets.load_dataset('mnist')

  train_images = mnist.train.images[:10000,:]
  train_labels = mnist.train.labels[:10000]

  model_fn \
    = lambda feature, target, mode : conv_model(feature, target, mode,
                                                image_dim=(28, 28),
                                                layer1_size=32,
                                                layer2_size=64,
                                                dense_layer_size=1024,
                                                num_classes=10)
  classifier = learn.Estimator(model_fn=model_fn)
  classifier.fit(train_images,
                 train_labels,
                 batch_size=100,
                 steps=20000)
  train_score = metrics.accuracy_score(train_labels,
                                       list(classifier.predict(train_images)))
  train_confusion_matrix = metrics.confusion_matrix(train_labels,
                                                    list(classifier.predict(train_images)))
  test_score = metrics.accuracy_score(mnist.test.labels,
                                      list(classifier.predict(mnist.test.images)))
  test_confusion_matrix = metrics.confusion_matrix(mnist.test.labels,
                                              list(classifier.predict(mnist.test.images)))
  print('Train Accuracy: {0:f}'.format(train_score))
  print('Test Accuracy: {0:f}'.format(test_score))
  print()
  print('Train Confusion Matrix:')
  print(train_confusion_matrix)
  print()
  print('Test Confusion Matrix:')
  print(test_confusion_matrix)


if __name__ == '__main__':
  tf.app.run()
