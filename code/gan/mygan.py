import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X =tf.placeholder(tf.float32, shape=[None, 784])

W1_d = tf.Variable(xavier_init([784, 128]))
B1_d = tf.Variable(xavier_init([128]))

W2_d = tf.Variable(xavier_init([128, 1]))
B2_d = tf.Variable(xavier_init([1]))

def discriminator(x):
    h1 = tf.nn.relu(tf.matmul(x, W1_d) + B1_d)
    logit = tf.matmul(h1, W2_d) + B2_d
    prob = tf.nn.sigmoid(logit)

    return prob, logit

W1_g = tf.Variable(xavier_init([100, 256]))
B1_g = tf.Variable(xavier_init([256]))

W2_g = tf.Variable(xavier_init([256, 784]))
B2_g = tf.Variable(xavier_init([784]))

def generator(z):
    h1 = tf.nn.relu(tf.matmul(z, W1_g) + B1_g)
    logit = tf.matmul(h1, W2_g) + B2_g
    prob = tf.nn.sigmoid(logit)
    return prob

Z = tf.placeholder(tf.float32, shape=[None, 100])
sample = generator(Z)
real_d, logit_real_d = discriminator(X)
gen_d, logit_gen_d = discriminator(sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real_d, labels=tf.ones_like(logit_real_d)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_gen_d, labels=tf.zeros_like(logit_gen_d)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_gen_d, labels=tf.ones_like(logit_gen_d)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=[W1_d, B1_d, W2_d, B2_d])
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=[W1_g, B1_g, W2_g, B2_g])

batch_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
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

def sample_Z(batch_size, Z_dim):
    return np.random.uniform(-1, 1, size=(batch_size, Z_dim))

if not os.path.exists('vanilla/'):
    os.mkdir('vanilla/')

i = 0
for it in range(100000):
    if it % 1000 == 0:
        samples = sess.run(sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('vanilla/mygan-{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_batch, _ = mnist.train.next_batch(batch_size)
    _, D_loss_cur = sess.run([D_solver, D_loss], feed_dict={X: X_batch, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_cur = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
    if it % 1000 == 0:
        print('Iter: {}, D_loss: {}, G_loss: {}'.format(it, D_loss_cur, G_loss_cur))
