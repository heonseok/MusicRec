#import unittest
import tensorflow as tf
import numpy as np
from network import Encoder

class TensorFlowTest(tf.test.TestCase):

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


if __name__ == '__main__':
    tf.test.main()
