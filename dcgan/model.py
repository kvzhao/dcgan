from __future__ import division
from __future__ import print_function

import os
import time
import math

from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from .ops import *
from .utils import *
from .base import BaseModel
from .data_handler import read_datasets
data = read_datasets()

# mnist test input data
#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('MNIST_data', one_hot=True)

class DCGAN(BaseModel):
    def __init__ (self, sess, config):
        super(DCGAN, self).__init__(config)

        self.sess = sess
        # batch normalization for discriminators
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # batch normalization for generators
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.build_model()

    def build_model(self):
        # image format: NHWC
        image_dim = [self.input_height, self.input_width, self.c_dim]
        # 1. placeholders
        self.inputs = tf.placeholder(
                tf.float32, [self.batch_size]+image_dim, name='real_images')
        self.sample_inputs = tf.placeholder(
                tf.float32, [self.sample_num]+image_dim, name='sample_inputs')
        inputs = self.inputs
        sample_inputs = self.sample_inputs

        # z
        self.z = tf.placeholder(
                tf.float32, [None, self.z_dim], name='z')
        # uniform distribution
        self.z_sum = histogram_summary('z', self.z)

        # 2. create G and D models
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # 3. summary
        self.d_sum = histogram_summary('d_real', self.D)
        self.d__sum = histogram_summary('d_fake', self.D_)
        self.G_sum = image_summary('G', self.G, max_outputs=6)


        # I maybe have to modify here
        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        self.d_loss_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary('d_loss_fake', self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
        self.d_loss_sum = scalar_summary('d_loss', self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def train(self):
        # Reading Data Phase
        # How to read data-inm read data into data_X

        # Setup Optimizer Phase
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        # Summary Records

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum,
            self.G_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./logs', self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load_model()
        if could_load:
            counter = checkpoint_counter
            print (' Load SUCCESS')
        else:
            print (' Load FAILED')

        # Training Process
        # epoch-loop
        for epoch in xrange(self.epoch):
            batch_idxs = min(len(data.images), self.train_size) // self.batch_size
            # batch-loop
            for idx in xrange(0, batch_idxs):
                batch_images, _ = data.next_batch(self.batch_size)
                sample_inputs, _ = data.next_batch(self.sample_num) # wired, but use now

                if(self.is_gray):
                    batch_images = np.array(batch_images).astype(np.float32)
                else:
                    batch_images = batch_images.astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                .astype(np.float32)

                # Update D Network
                _, summary_str = self.sess.run(
                        [d_optim, self.d_sum],
                        feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                # Update G Network
                for _ in range(2):
                    # Run g_optim twice
                    # to make sure that d_loss does not go to zero
                    _, summary_str = self.sess.run(
                            [g_optim, self.g_sum],
                            feed_dict={self.z:  batch_z})
                    self.writer.add_summary(summary_str, counter)
                # error function evalutaion
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG      = self.g_loss.eval({self.z: batch_z})

                counter += 1
                if np.mod(counter, self.display_status_per_iter) == 1:
                    print ('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' \
                        % (epoch, idx, batch_idxs, time.time()-start_time, errD_fake+errD_real, errG))

                if np.mod(counter, self.sample_per_iter) == 1:
                    # sample (testing) and save data out
                    samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.inputs: sample_inputs,
                                self.z: sample_z
                                }
                            )
                    # Save images
                    save_images(samples, [16, 16],
                            './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, self.save_per_iter) == 1:
                    # backup model
                    self.save_model(counter)


    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h3), h3

    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            # height and width of images divided by 2 in each layer
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # project z and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim*4*s_h8*s_w8, scope='g_h0_lin', with_w=True)
            self.h0 = tf.reshape(
                    self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            # conv1
            h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(h1))

            # conv2
            h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            # conv3, final layer
            h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3', with_w=True)
            h3 = tf.nn.tanh(h3)

            return h3

    def sampler(self, z):
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            h0 = tf.reshape(
                    linear(z, self.gf_dim*4*s_h8*s_w8, 'g_h0_lin'),
                    [-1, s_h8, s_w8, self.gf_dim*4])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            # conv1
            h1 = deconv2d(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            # conv2
            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            # conv3, final layer
            h3 = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')
            h3 = tf.nn.tanh(h3)

            return h3
