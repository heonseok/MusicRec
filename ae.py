import tensorflow as tf

from network import Encoder
from network import Decoder

class AE:
    def __init__(self, logger, learning_rate, input_dim, z_dim, ae_h_dims, *args, **kwargs):
        self.scope = "AE"

        self.logger = logger
        self.learning_rate = learning_rate

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.enc_layer_dims = [input_dim, *ae_h_dims, z_dim]
        self.dec_layer_dims = [z_dim, *list(reversed(ae_h_dims)), input_dim] #todo : just reverse enc_layer_dims

        self.logger.info("[*] Building AE model")

        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(tf.float32, [None, self.input_dim])

            enc = Encoder(self.enc_layer_dims)
            dec = Decoder(self.dec_layer_dims)

            z_layer = enc.encode(self.input)
            # todo : how to handle output?
            _, _, self.output = dec.decode(z_layer)

            # todo: refactoring get theta method --> get solver?
            enc_theta = enc.get_theta()
            dec_theta = dec.get_theta()
            self.theta = [*enc_theta, *dec_theta]

            #l2_loss = enc.get_l2_loss() +
            self.recon_loss = tf.reduce_mean(tf.square(self.input-self.output))
            self.solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.recon_loss, var_list=theta)

    def train(self, train_data, epoch, batch_size, sess):
        sess.run(tf.global_variables_initializer())
        self.logger.info("[*] Start training")
        num_data = train_data.shape[0]
        total_batch = int(num_data/batch_size)

        for epoch_idx in range(epoch):
            for batch_idx in range(total_batch):
                batch_input = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
                _, recon_loss = sess.run([self.solver, self.recon_loss], feed_dict=self.theta)

