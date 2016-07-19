import tensorflow as tf



def training_mse_loss(output, labels):
    with tf.variable_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(tf.sub(output, labels)), name='mse_loss')
    return loss

def training_sigmoid_cross_entropy(output, labels):
    with tf.variable_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(output, labels, name='sigmoid_cross_entropy')
    return loss

def training_op(loss, initial_learning_rate, global_step, decay_steps, decay_rate):
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
        name='learning_rate'
    )

    optimizer       = tf.train.GradientDescentOptimizer(lr)
    gradients       = optimizer.compute_gradients(loss)
    gradients       = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in gradients]
    apply_gradients = optimizer.apply_gradients(gradients, global_step=global_step)

    with tf.control_dependencies([apply_gradients]):
        train_op = tf.no_op(name='train')

    return train_op, gradients
