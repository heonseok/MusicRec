from __future__ import absolute_import

import tensorflow as tf
import logging
import os

from ae import AE
from load_data import load_data

### ===== Set FLAGS ===== ###
flags = tf.app.flags
flags.DEFINE_string("model", "VAE_GAN", "Model to run [AE, VAE, VAE_GAN]")

flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate [0.0001]")
flags.DEFINE_integer("batch_size", 512, "Batch size [512]")
flags.DEFINE_integer("batch_logging_step", 100, "Logging step for batch [100]")
flags.DEFINE_integer("epoch_logging_step", 1, "Logging step for epoch [1]")  # Need?

flags.DEFINE_integer("input_dim", 24000, "Dimension of input [24000]")
flags.DEFINE_string("ae_h_dim_list", "[2048]", "List of AE dimensions [2048]")
flags.DEFINE_integer("z_dim", 128, "Dimension of z [128]")
flags.DEFINE_string("dis_h_dim_list", "[2048]", "List of Discriminator dimension [2048]")

flags.DEFINE_integer("gpu_id", 0, "GPU id [0]")
flags.DEFINE_string("data_dir", "data", 'Directory name to load input data [data]')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the logs [log]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [Fasle]")
flags.DEFINE_boolean("continue_train", None,
                     "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")
FLAGS = flags.FLAGS

### ===== Set logger ===== ###
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
fh = logging.FileHandler(FLAGS.log_dir)

fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
sh.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(sh)
logger.addHandler(fh)


def main(_):
    #print(FLAGS.__flags)
    file_name =  'm[' + FLAGS.model + ']_lr[' + str(FLAGS.learning_rate) + ']_b[' + str(FLAGS.batch_size) + \
                 ']_ae' + FLAGS.ae_h_dim_list + '_z[' + str(FLAGS.z_dim) +  ']_dis' + FLAGS.dis_h_dim_list
    logger.info(file_name)

    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        ### ===== Build model ===== ###
        if FLAGS.model == "AE":
            logger.info("Build AE model")
            model = AE(logger, FLAGS.learning_rate, FLAGS.input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list))

        elif FLAGS.model == "VAE":
            logger.info("Build VAE model")

        elif FLAGS.model == "VAE_GAN":
            logger.info("Build VAE_GAN model")


        ### ===== Train/Test =====###

        if FLAGS.is_train:
            #logger.info("Start training")
            train_data = load_data(os.path.join(FLAGS.data_dir, 'train_data.npy'))
            val_data = load_data(os.path.join(FLAGS.data_dir, 'val_data.npy'))
            #print(train_data.shape)
            model.train(train_data, FLAGS.batch_size)
        else:
            logger.info("Start testing")
            test_data = load_data(os.path.join(FLAGS.data_dir, 'test_data.npy'))


if __name__ == '__main__':
    tf.app.run()