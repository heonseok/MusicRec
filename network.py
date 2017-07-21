import tensorflow as tf

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    xavier_stddev = tf.cast(xavier_stddev, tf.float32)
    #print(xavier_stddev.dtype)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def set_fc_layer(layer_dims):
    weights = []
    biases = []
    for i in range(len(layer_dims)-1):
        weights.append(tf.Variable(xavier_init([layer_dims[i], layer_dims[i+1]]), name = "W%d" % (i+1))) #, dtype="float32" ))
        biases.append(tf.Variable(tf.random_normal([layer_dims[i+1]]), name = "b%d" % (i+1))) #, dtype="float32" ))
    return weights, biases

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class BasicNet():
    theta = []
    layer_dims = []

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.weights, self.biases = set_fc_layer(self.layer_dims)
        self.theta = [*self.weights, *self.biases]
        self.dropout_keep_prob = tf.constant(0.5)

    def get_theta(self):
        return self.theta

class Decoder(BasicNet):
    def __init__(self, *args, **kwargs):
        with tf.variable_scope('decoder'):
            super(Decoder, self).__init__(*args, **kwargs)

    def decode(self, previous_layer):
        for i in range(len(self.layer_dims)-2):
            previous_layer = lrelu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])
            #previous_layer = tf.nn.relu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])
            # drop_out for regularization
            previous_layer = tf.nn.dropout(previous_layer, self.dropout_keep_prob)

        logits = tf.matmul(previous_layer, self.weights[-1]) + self.biases[-1]
        prob = tf.nn.sigmoid(logits)
        dec_val = tf.nn.tanh(logits) # decoder value
        return prob, logits, dec_val


class Encoder(BasicNet):
    def __init__(self, *args, **kwargs):
        with tf.variable_scope('encoder'):
            super(Encoder, self).__init__(*args, **kwargs)

    def set_distribution_layer(self, z_dim):
        self.W_mu = tf.Variable(xavier_init([self.layer_dims[-1], z_dim]))
        self.b_mu = tf.Variable(tf.random_normal([z_dim]))

        self.W_sigma = tf.Variable(xavier_init([self.layer_dims[-1], z_dim]))
        self.b_sigma = tf.Variable(tf.random_normal([z_dim]))

        self.theta = [*self.weights, *self.biases, self.W_mu, self.b_mu, self.W_sigma, self.b_sigma]

    def encode_distribution(self, previous_layer):
        for i in range(len(self.layer_dims)-1):
            previous_layer = lrelu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])
            #previous_layer = tf.nn.relu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])

        z_mu = tf.matmul(previous_layer, self.W_mu) + self.b_mu
        z_logvar = tf.matmul(previous_layer, self.W_sigma) + self.b_sigma

        return z_mu, z_logvar

    def encode(self, previous_layer):
        for i in range(len(self.layer_dims)-1):
            previous_layer = lrelu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])
            #previous_layer = tf.nn.relu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])

        return previous_layer


class Discriminator(BasicNet):
    def __init__(self, *args, **kwargs):
        with tf.variable_scope('discriminator'):
            super(Discriminator, self).__init__(*args, **kwargs)

    def discriminate(self, previous_layer):
        for i in range(len(self.layer_dims)-2):
            previous_layer = lrelu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])
            #previous_layer = tf.nn.relu(tf.matmul(previous_layer, self.weights[i]) + self.biases[i])

        logits = tf.matmul(previous_layer, self.weights[-1]) + self.biases[-1]
        prob = tf.nn.sigmoid(logits)
        return prob, logits