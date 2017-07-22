#import unittest
import tensorflow as tf
import numpy as np
from network import Encoder

class TensorFlowTestCase(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
            x = tf.square([2,3])
            x = tf.Print(x, [x], message="Square")

            self.assertAllEqual(x.eval(), [4,9])

    def test_init_variable_with_zeros(self):
        with self.test_session() as sess:
            x = tf.Variable(tf.zeros([1,3]), name='var')
            x = tf.Print(x, [x], message="Zero init")

            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(x.eval(), np.reshape(np.zeros(3), [1,3]))

    def test_mean_squared_error(self):
        with self.test_session() as sess:
            a = tf.constant([3.0, 4.0, 0.0, 3.0], shape=[2,2])
            b = tf.constant([0.0, 0.0, 4.0, 0.0], shape=[2,2])
            mse = tf.reduce_mean(tf.square(a-b))
            mse_ = tf.losses.mean_squared_error(a,b)
            self.assertEqual(12.5, mse.eval())
            self.assertEqual(12.5, mse_.eval())


if __name__ == '__main__':
    tf.test.main()
