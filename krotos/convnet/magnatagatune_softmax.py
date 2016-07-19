import tensorflow as tf
import numpy as np
import time
import datetime

from krotos.convnet.model.network import build_network, sigmoid_layer
from krotos.convnet.model.training import training_sigmoid_cross_entropy, training_op
from krotos.magnatagatune import Dataset
from krotos.debug import report
from krotos.paths import PATHS



MAX_BATCHES         = 2300
SAVE_BATCHES        = 50
BATCH_SIZE          = 10

DATASET             = Dataset(new=False, training_split=0.7, validation_split=0.1, testing_split=0.2)
CHECKPOINT_FILENAME = 'model.ckpt'

MAPPING_DIM         = 188

IMAGE_HEIGHT        = 625
IMAGE_WIDTH         = 128

INITIAL_LEARNING_RATE       = 0.001
LEARNING_RATE_DECAY_FACTOR  = 0.96
DECAY_STEPS                 = 200



def batch():
    return zip(*DATASET.minibatch(samples=BATCH_SIZE))

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        batch_images    = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH], name='spectrogram_image')
        batch_labels    = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAPPING_DIM], name='labels')

        output                          = sigmoid_layer(build_network(batch_images, output_size=MAPPING_DIM))
        sigmoid_cross_entropy_loss_op   = training_sigmoid_cross_entropy(output, batch_labels)
        train_op, gradients             = training_op(
            loss=sigmoid_cross_entropy_loss_op,
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

        ckpt = tf.train.get_checkpoint_state(PATHS['convnet_dir'] + 'magnatagatune_softmax/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        tf.train.start_queue_runners(sess)

        for step in xrange(sess.run(global_step) + 1, MAX_BATCHES + 1):
            images, labels = batch()

            time_start_world    = time.time()
            time_start_proc     = time.clock()

            mean_loss, _, _ = sess.run([tf.reduce_mean(sigmoid_cross_entropy_loss_op), train_op, check_op], feed_dict={
                batch_images: images,
                batch_labels: labels,
            })

            report('Training: Step {0} had a mean sigmoid cross entropy loss {1}'.format(step, mean_loss))
            report('\t{0}'.format(datetime.datetime.now()))
            report("\tSession ran in {}s ({}s process time).".format(time.time() - time_start_world, time.clock() - time_start_proc))

            if step % SAVE_BATCHES == 0:
                saver.save(sess, PATHS['convnet_dir'] + 'magnatagatune_softmax/' + CHECKPOINT_FILENAME, global_step=step)

        sess.close()
