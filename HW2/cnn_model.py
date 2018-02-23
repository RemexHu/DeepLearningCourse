import tensorflow as tf
import time
import numpy as np


class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, index, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling_%d' % index):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class batch_norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class Local_Response_Norm_layer(object):
    def __init__(self, input_x):
        with tf.variable_scope('local_response_norm'):
            cell_out = tf.nn.local_response_normalization(input_x,
                                                          depth_radius=5,
                                                          bias=1,
                                                          alpha=1,
                                                          beta=0.5,
                                                          name=None)
            self.cell_out = cell_out
    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out



def LeNet(input_x, input_y,
          img_len=24, channel_num=3, output_size=10,
          conv_featmap=[48, 48, 96, 96, 192, 192], fc_units=[512, 256],
          conv_kernel_size=[3, 3, 3, 3, 3, 3], pooling_size=[2, 2, 2],
          l2_norm=0.01, seed=235):
    """
        LeNet is an early and famous CNN architecture for image classfication task.
        It is proposed by Yann LeCun. Here we use its architecture as the startpoint
        for your CNN practice. Its architecture is as follow.

        input >> Conv2DLayer >> Conv2DLayer >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        Or

        input >> [conv2d-maxpooling] >> [conv2d-maxpooling] >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        http://deeplearning.net/tutorial/lenet.html

    """

    #assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,
                              index=0)

    conv_layer_01 = conv_layer(input_x=conv_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed,
                              index=1)


    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_01.output(),
                                        k_size=pooling_size[0],
                                        index=0,
                                        padding="SAME")

    dropout_0 = tf.nn.dropout(pooling_layer_0.output(), keep_prob=0.25)

    batch_norm_layer_0 = batch_norm_layer(input_x=dropout_0)


    #LRN_layer_0 = Local_Response_Norm_layer(input_x=pooling_layer_0.output())



    #conv layer
    conv_layer_1 = conv_layer(input_x=batch_norm_layer_0.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              rand_seed=seed,
                              index=2)

    conv_layer_11 = conv_layer(input_x=conv_layer_1.output(),
                              in_channel=conv_featmap[2],
                              out_channel=conv_featmap[3],
                              kernel_shape=conv_kernel_size[3],
                              rand_seed=seed,
                              index=3)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_11.output(),
                                        k_size=pooling_size[1],
                                        index=1,
                                        padding="SAME")

    #LRN_layer_1 = Local_Response_Norm_layer(input_x=pooling_layer_1.output())

    #dropout_1 = tf.nn.dropout(pooling_layer_1.output(), keep_prob=0.25)

    batch_norm_layer_1 = batch_norm_layer(input_x=pooling_layer_1.output())

    # conv layer
    conv_layer_2 = conv_layer(input_x=batch_norm_layer_1.output(),
                              in_channel=conv_featmap[3],
                              out_channel=conv_featmap[4],
                              kernel_shape=conv_kernel_size[4],
                              rand_seed=seed,
                              index=4)

    conv_layer_21 = conv_layer(input_x=conv_layer_2.output(),
                              in_channel=conv_featmap[4],
                              out_channel=conv_featmap[5],
                              kernel_shape=conv_kernel_size[5],
                              rand_seed=seed,
                              index=5)


    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_21.output(),
                                        k_size=pooling_size[2],
                                        index=2,
                                        padding="SAME")

    #LRN_layer_2 = Local_Response_Norm_layer(input_x=pooling_layer_2.output())

    #dropout_2 = tf.nn.dropout(pooling_layer_2.output(), keep_prob=0.25)

    batch_norm_layer_2 = batch_norm_layer(input_x=pooling_layer_2.output())



    # flatten
    pool_shape = batch_norm_layer_2.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(batch_norm_layer_2.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)


    # dropout layer
    dropout_3 = tf.nn.dropout(fc_layer_0.output(), keep_prob=0.5)

    batch_norm_layer_3 = batch_norm_layer(input_x=dropout_3)

    fc_layer_1 = fc_layer(input_x=batch_norm_layer_3.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1)
    dropout_4 = tf.nn.dropout(fc_layer_1.output(), keep_prob=0.5)

    batch_norm_layer_4 = batch_norm_layer(input_x=dropout_4)

    #fc layer
    fc_layer_2 = fc_layer(input_x=batch_norm_layer_4.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)


    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight]
    fc_w = [fc_layer_0.weight, fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_2.output(), loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss):
    with tf.name_scope('train_step'):
        """
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=100,
                                                   decay_rate=0.9,
                                                   staircase=True)
        """
        #step = tf.train.AdamOptimizer(0.0009).minimize(loss)
        step = tf.train.AdamOptimizer(0.0008).minimize(loss)
        #tf.summary.scalar('learning_rate', loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num





# training function for the LeNet model
def training(X_train, y_train, X_val, y_val,
             conv_featmap=[48, 48, 96, 96, 192, 192],
             fc_units=[512, 256],
             conv_kernel_size=[3, 3, 3, 3, 3, 3],
             pooling_size=[2, 2, 2],
             l2_norm=0.01,
             seed=235,
             learning_rate=0.001,
             epoch=2000,
             batch_size=100,
             verbose=False,
             pre_trained_model=None):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = LeNet(xs, ys,
                         img_len=32,
                         channel_num=3,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))



    step = train_step(loss)
    eve = evaluate(output, ys)

    dir = "/home/runchen/Downloads/CNN_Tutorial/Tutorial/"
    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))


            perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[perm, :, :, :]
            y_train = y_train[perm]


            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    tf.summary.scalar("valid_acc", valid_acc)
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))



    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))