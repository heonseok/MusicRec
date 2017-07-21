import tensorflow as tf
from network import set_fc_layer

class AE:
    def __init__(self, logger, learning_rate, X, X_dim, z_dim, ae_h_dims, *args, **kwargs):
        self.logger = logger
        self.learning_rate = learning_rate
        self.scope = "AE"

        self.X = X
        self.z_dim = z_dim
        enc_layer_dims = [X_dim, *ae_h_dims, z_dim]
        dec_layer_dims = [z_dim, *list(reversed(ae_h_dims)), X_dim]

        self.build_model()


    def build_model(self):
        self.logger.info("[*] Building AE model")

        with tf.variable_scope(self.scope):

