import tensorflow.compat.v1 as tf

#### Create tf model for Client ####

def GATTA_Net(input_shape, num_classes, learning_rate, graph, Merge):
    """
        Construct the GATTA model.
        input_shape: The shape of input (`list` like)
        num_classes: The number of output classes (`int`)
        learning_rate: learning rate for optimizer (`float`)
        graph: The tf computation graph (`tf.Graph`)
        local_paras: shape of [1, 1, len=out_sz]
        neigh_paras: [1, num_neig, len=out_sz]
    """
    with graph.as_default():
        X = tf.placeholder(tf.float32, input_shape, name='X')
        Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')

        conv1 = conv(X, 5, 5, 64, 1, 1, name='conv1')
        norm1 = lrn(conv1, 4, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 64, 1, 1, name='conv2')
        norm2 = lrn(conv2, 4, 1e-04, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

        fc1 = fc_layer(flattened, 7 * 7 * 64,  384, name='fc1')
        fc2 = fc_layer(fc1,  384,  192, name='fc2')

        lp = tf.placeholder(tf.float32, [1, 1, 192*num_classes+num_classes], name='Local_p2')
        np = tf.placeholder(tf.float32, [1, None, 192*num_classes+num_classes], name='Neig_p2')

        logits_ns, para = fc_layer_ns(fc2, lp, np, 192, num_classes, elu=False, name='fc2_ns', name2='GAT2')

        # loss and optimizer
        loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ns, labels=Y))

        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=learning_rate)

        # optimizer = AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        prediction = tf.nn.softmax(logits_ns)
        pred = tf.argmax(prediction, 1)

        # accuracy
        correct_pred = tf.equal(pred, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))

        return X, Y, train_op, loss_op, accuracy, para, lp, np


def conv(x, filter_height, filter_width, num_filters,
         stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(
        i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[
                                      filter_height, filter_width,
                                      int(input_channels / groups), num_filters
                                  ])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3,
                                 num_or_size_splits=groups,
                                 value=weights)
        output_groups = [
            convolve(i, k) for i, k in zip(input_groups, weight_groups)
        ]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    elu = tf.nn.elu(bias, name=scope.name)

    return elu


def fc_layer(x, input_size, output_size, name, elu=True, k=20):
    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape=[input_size, output_size])
        b = tf.get_variable('biases', shape=[output_size])
        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if elu:
        # Apply ReLu non linearity.
        a = tf.nn.elu(z)
        return a

    else:
        return z


def max_pool(x,
             filter_height, filter_width,
             stride_y, stride_x,
             name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool2d(x,
                            ksize=[1, filter_height, filter_width, 1],
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding,
                            name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

def dropout(x, rate):
    """Create a dropout layer."""
    return tf.nn.dropout(x, rate=rate)



def fc_layer_ns(x, local_paras, neigh_paras, input_size, output_size, name, name2, elu=True):
    """Create a node-specific fully connected layer."""
    mu = 0.9
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape=[input_size, output_size])
        b = tf.get_variable('biases', shape=[output_size])
        # Matrix multiply weights and inputs and add biases.
        parameters = GAT_layer(local_paras, neigh_paras, input_size*output_size+output_size, name2=name2)
        W1 = parameters[0, 0, 0: input_size*output_size]
        W_add = mu* W +  (1-mu)*tf.reshape(W1, [input_size, output_size])
        b1 =  parameters[0, 0, input_size*output_size: input_size*output_size+output_size]
        b_add =  mu*b +  (1-mu)*tf.reshape(b1, [output_size])
        z = tf.nn.bias_add(tf.matmul(x, W_add), b_add, name=scope.name)

    if elu:
        # Apply ReLu non linearity.
        a = tf.nn.elu(z)
        return a, parameters

    else:
        return z, parameters


def GAT_layer(local_paras, neigh_paras, out_sz, name2, in_drop=0.4, coef_drop=0.0, residual=False, activation=tf.nn.elu):
    # local_paras [1, 1, len=out_sz]
    # neigh_paras [1, num_neig, len=out_sz]
    with tf.variable_scope(name2) as scope:
        # seq_local = tf.layers.conv1d(local_paras, out_sz, 1, use_bias=False)
        # seq_neig = tf.layers.conv1d(all_paras, out_sz, 1, use_bias=False)
        seq_neig = neigh_paras
        # all_paras = tf.concat([local_paras, neigh_paras], 1)
        f_1 = tf.layers.conv1d(local_paras, 1, 1) # [1, 1, 1]
        f_2 = tf.layers.conv1d(seq_neig, 1, 1) # [1, neigh, 1]
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.elu(logits))

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            all_paras = tf.nn.dropout(seq_neig, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_neig) # 得到[1, 1, sz]的tensor
        biase = tf.get_variable('biases_GAT', shape=[out_sz])
        ret = tf.nn.bias_add(vals, biase)

        # residual connection
        if residual:
            if local_paras.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(local_paras, ret.shape[-1], 1)  # activation
            else:
                ret = ret + local_paras

        return activation(ret)  #[1, 1, sz]
