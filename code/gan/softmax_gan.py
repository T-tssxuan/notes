import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

batch_size = 10

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def conv2d(x, w_conv):
    return tf.nn.conv2d(x, w_conv, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_conv(x, shape, prefix):
    w_conv = tf.get_variable(prefix + '_w_conv', shape=shape, initializer=tf.random_normal_initializer(0, 0.01))
    b_conv = tf.get_variable(prefix + '_b_conv', shape=shape[-1:], initializer=tf.constant_initializer(0.01))
    h_conv = tf.nn.relu(conv2d(x, w_conv) + b_conv)
    return max_pool_2x2(h_conv)

def discriminator(x, keep_prob, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.variable_scope('d_conv1'):
            h_conv1 = get_conv(x, [5, 5, 1, 32], 'd')
        with tf.variable_scope('d_conv2'):
            h_conv2 = get_conv(h_conv1, [5, 5, 32, 64], 'd')
        with tf.variable_scope('d_conv3'):
            h_conv3 = get_conv(h_conv2, [5, 5, 64, 128], 'd')

        w_fc1 = tf.get_variable('d_w_fc1', shape=[4*4*128, 128], initializer=tf.random_normal_initializer(0, 0.01))
        b_fc1 = tf.get_variable('d_b_fc1', shape=[128], initializer=tf.constant_initializer(0.))
        
        h_conv3_flat = tf.reshape(h_conv3, [-1, 4 * 4 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = tf.get_variable('d_w_fc2', shape=[128, 1], initializer=tf.random_normal_initializer(0, 0.01))
        b_fc2 = tf.get_variable('d_b_fc2', shape=[1], initializer=tf.constant_initializer(0.))

        return tf.matmul(h_fc1_drop, w_fc2) + b_fc2

def d_conv2d(x, w_d_conv, output_shape):
    return tf.nn.conv2d_transpose(x, w_d_conv, output_shape, strides=[1, 2, 2, 1], padding='SAME')

def get_d_conv(x, shape, output_shape, prefix):
    w_d_conv = tf.get_variable(prefix + '_w_d_conv', shape=shape, initializer=tf.random_normal_initializer(0, 0.01))
    b_d_conv = tf.get_variable(prefix + '_b_d_conv', shape=[shape[2]], initializer=tf.constant_initializer(0.01))
    h_d_conv = d_conv2d(x, w_d_conv, output_shape)
    return tf.nn.relu(h_d_conv + b_d_conv)

def generator(x):
    with tf.variable_scope('generator'):
        w_fc1 = tf.get_variable('g_w_fc1', shape=[100, 4 * 4 * 128])
        b_fc1 = tf.get_variable('g_b_fc1', shape=[4 * 4 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
        g_x = tf.reshape(h_fc1, [-1, 4, 4, 128])
        print(g_x.shape)
        with tf.variable_scope('g_d_conv1'):
            h_d_conv1 = get_d_conv(g_x, [3, 3, 64, 128], [batch_size, 7, 7, 64], 'g')
            print(h_d_conv1.shape)
        with tf.variable_scope('g_d_conv2'):
            h_d_conv2 = get_d_conv(h_d_conv1, [3, 3, 32, 64], [batch_size, 14, 14, 32], 'g')
            print(h_d_conv2.shape)
        with tf.variable_scope('g_d_conv3'):
            h_d_conv3 = get_d_conv(h_d_conv2, [5, 5, 1, 32], [batch_size, 28, 28, 1], 'g')
        print(h_d_conv3.shape)
        return h_d_conv3

def get_data(size):
    return np.random.uniform(-1, 1, size=size)

data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Z = tf.placeholder(tf.float32, shape=[None, 100])
drop_prob = tf.placeholder(tf.float32)

G_sample = generator(Z)

D_logit_real = discriminator(data, drop_prob)
D_logit_fake = discriminator(G_sample, drop_prob, reuse=True)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake), logits=D_logit_fake))

d_loss = D_loss_real + D_loss_fake

g_var_list = [var for var in tf.global_variables() if var.name.startswith('generator')]
print([var.name for var in g_var_list])
d_var_list = [var for var in tf.global_variables() if var.name.startswith('discriminator')]
print([var.name for var in d_var_list])

D_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_var_list)
G_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in range(10000):
    x, _ = mnist.train.next_batch(batch_size)
    x = x.reshape([-1, 28, 28, 1])
    z = get_data([batch_size, 100])
    dl, dlr, dlf, _ = sess.run([d_loss, D_loss_real, D_loss_fake, D_solver], feed_dict={data: x, Z: z, drop_prob: 0.5})

    z = get_data([batch_size, 100])
    # print(z)
    _, gl = sess.run([G_solver, g_loss], feed_dict={Z: z, drop_prob: 1})

    if i % 10 == 0:
        logging.info('g_loss: {}, d_loss: {}'.format(dl, gl))
        logging.info('D_loss_real: {}, D_loss_fake: {}'.format(dlr, dlf))

        z = get_data([batch_size, 100])
        samples = sess.run(G_sample, feed_dict={Z: z})
        fig = plot(samples[:16])
        if not os.path.isdir('rst/'):
            os.mkdir('rst/')
        plt.savefig('rst/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')

