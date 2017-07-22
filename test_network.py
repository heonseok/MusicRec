#import unittest
import tensorflow as tf
import numpy as np
from network import Encoder

class NetworkTestCase(tf.test.TestCase):

    def test_encoder_output_shape(self):
        with self.test_session():

            input_dim = 100;
            output_dim = 10;
            batch_size = 512;

            layer_dims = [input_dim, 50, output_dim]

            X = tf.constant(np.random.random(batch_size*input_dim), shape=[batch_size, input_dim], dtype=tf.float32)
            enc = Encoder(layer_dims)

            self.assertShapeEqual(np.reshape(np.zeros(batch_size*output_dim), [batch_size, output_dim]), enc.encode(X))

    def test_decoder_output_shape(self):
        with self.test_session():

            input_dim = 10;
            output_dim = 100;
            batch_size = 1;

            layer_dims = [input_dim, 50, output_dim]

            X = tf.constant(np.random.random(batch_size*input_dim), shape=[batch_size, input_dim], dtype=tf.float32)
            enc = Encoder(layer_dims)

            self.assertShapeEqual(np.reshape(np.zeros(batch_size*output_dim), [batch_size, output_dim]), enc.encode(X))


if __name__ == '__main__':
    tf.test.main()
