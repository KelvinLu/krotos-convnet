import tensorflow as tf



def get_variable(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def get_variable_normdist(name, shape, stddev):
    return get_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev))

def get_variable_zeroes(name, shape):
    return get_variable(name, shape, tf.constant_initializer(0.0))

def audio_conv_layer(name, input_layer, features, pooling):
    with tf.variable_scope(name) as scope:
        input_features = input_layer.get_shape().dims[3].value

        kernel  = get_variable_normdist('kernel', shape=[1, 4, input_features, features], stddev=1e-4)
        bias    = get_variable_zeroes('bias', shape=[features])

        conv_layer  = tf.nn.bias_add(tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME'), bias)
        relu_layer  = tf.nn.relu(conv_layer)
        pool_layer  = tf.nn.max_pool(relu_layer, ksize=[1, 1, pooling, 1], strides=[1, 1, pooling, 1], padding='SAME', name='output')

    return pool_layer

def fc_layer(name, input_layer, neurons, dropout=0.5):
    with tf.variable_scope(name) as scope:
        input_n = input_layer.get_shape().dims[1].value

        weights = get_variable_normdist('weights', shape=[input_n, neurons], stddev=4e-2)
        bias    = get_variable_zeroes('bias', shape=[neurons])

        perceptron_layer    = tf.add(tf.matmul(input_layer, weights), bias)
        output_layer        = tf.nn.dropout(perceptron_layer, keep_prob=dropout, name='output')

    return output_layer

def network(input_layer):
    with tf.variable_scope('network') as scope:
        # Use expand_dims to convert tensor of [batch, height, width]
        # to tensor of [batch=batch, height=1, width=height, features=width]
        channeled_input = tf.expand_dims(input_layer, 1)

        # Convolutional layers
        conv_1 = audio_conv_layer(
            name='conv_1',
            input_layer=channeled_input,
            features=256,
            pooling=4
        )

        conv_2 = audio_conv_layer(
            name='conv_2',
            input_layer=conv_1,
            features=256,
            pooling=2
        )

        conv_3 = audio_conv_layer(
            name='conv_3',
            input_layer=conv_2,
            features=512,
            pooling=2
        )

        # Temporal-global pooling
        global_pool_width   = conv_3.get_shape().dims[2].value

        global_pool_mean    = tf.squeeze(tf.nn.avg_pool(
            conv_3,
            ksize=[1, 1, global_pool_width, 1],
            strides=[1, 1, global_pool_width, 1],
            padding='SAME'
        ))

        global_pool_max     = tf.squeeze(tf.nn.max_pool(
            conv_3,
            ksize=[1, 1, global_pool_width, 1],
            strides=[1, 1, global_pool_width, 1],
            padding='SAME'
        ))

        global_pool_l2norm  = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(conv_3), 2)))

        global_pooling = tf.concat(1, [global_pool_mean, global_pool_max, global_pool_l2norm])

        # Fully-connected layers
        fc_1    = fc_layer(
            name='fc_1',
            input_layer=global_pooling,
            neurons=2048,
        )

        fc_2    = fc_layer(
            name='fc_2',
            input_layer=fc_1,
            neurons=2048,
        )

        fc_3    = fc_layer(
            name='fc_3',
            input_layer=fc_2,
            neurons=50,
        )

    return fc_3

def network_loss(output, labels):
    with tf.variable_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(tf.sub(output, labels)), 1, name='batch_monotonic_l2_loss')
    return loss
