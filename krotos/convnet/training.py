import tensorflow as tf
import numpy as np
import time
import datetime

from krotos.convnet.model.network import build_network
from krotos.convnet.model.training import training_mse_loss, training_op
from krotos.msd.dataset import Dataset
from krotos.debug import report
from krotos.paths import PATHS



MAX_BATCHES         = 100000
SAVE_BATCHES        = 10
BATCH_SIZE          = 10

DATASET             = Dataset.instance(new=False, training_split=0.7, validation_split=0.1, testing_split=0.2)
CHECKPOINT_FILENAME = 'model.ckpt'

MAPPING             = 'latent_features'
MAPPING_DIM         = 50

IMAGE_HEIGHT        = 625
IMAGE_WIDTH         = 128

INITIAL_LEARNING_RATE       = 0.001
LEARNING_RATE_DECAY_FACTOR  = 0.05
DECAY_STEPS                 = 4000



def batch():
    minibatch = DATASET.minibatch(
        n=BATCH_SIZE,
        mapping=MAPPING,
        trim=True,
        normalize=True
    )

    images, labels = zip(*minibatch)

    return images, labels

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        batch_images    = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH], name='spectrogram_image')
        batch_labels    = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAPPING_DIM], name='labels')

        output              = build_network(batch_images, output_size=MAPPING_DIM)
        mse_loss_op         = training_mse_loss(output, batch_labels)
        train_op, gradients = training_op(
            loss=mse_loss_op,
            global_step=global_step,
            initial_learning_rate=INITIAL_LEARNING_RATE,
            decay_rate=LEARNING_RATE_DECAY_FACTOR,
            decay_steps=DECAY_STEPS
        )

        check_op = tf.add_check_numerics_ops()

        saver       = tf.train.Saver(tf.all_variables())

        init_op     = tf.initialize_all_variables()

        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(PATHS['convnet_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        tf.train.start_queue_runners(sess)

        for step in xrange(sess.run(global_step) + 1, MAX_BATCHES + 1):
            images, labels = batch()

            time_start_world    = time.time()
            time_start_proc     = time.clock()

            mse_loss, _, _ = sess.run([mse_loss_op, train_op, check_op], feed_dict={
                batch_images: images,
                batch_labels: labels,
            })

            report('Training: Step {0} had MSE loss {1}'.format(step, mse_loss))
            report('\t{0}'.format(datetime.datetime.now()))
            report("\tSession ran in {}s ({}s process time).".format(time.time() - time_start_world, time.clock() - time_start_proc))

            if step % SAVE_BATCHES == 0:
                saver.save(sess, PATHS['convnet_dir'] + CHECKPOINT_FILENAME, global_step=step)

        sess.close()
