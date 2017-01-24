"""Convolutional Neural Net using tflearn."""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def conv_network(image_dim, num_classes):
  """Build a CNN for the given image dimensions.

  A single channel is assumed, so image_dim = (m, n).
  """
  network = input_data(shape=[None, image_dim[0], image_dim[1], 1],
                       name='input')
  network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = local_response_normalization(network)
  network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = local_response_normalization(network)
  network = fully_connected(network, 128, activation='tanh')
  network = dropout(network, 0.8)
  network = fully_connected(network, 256, activation='tanh')
  network = dropout(network, 0.8)
  network = fully_connected(network, num_classes, activation='softmax')
  return regression(network, optimizer='adam', learning_rate=0.01,
                    loss='categorical_crossentropy', name='target')
