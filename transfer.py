"""
    Copyright 2018 Ashar <ashar786khan@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from vgg19 import ImageNetVGG19

warnings.simplefilter('ignore', category=FutureWarning)

class NeuralStyleTransfer:
    def __init__(self,
                 content_pth,
                 style_pth,
                 gen_file_name='./styled_image.jpg',
                 epochs=10000,
                 initialize_randomly=False,
                 learning_rate=0.5,
                 content_factor=0.001,
                 style_factor=1,
                 style_pooling_layers=[3, 4],
                 style_layers_weight=[0.5, 0.5],
                 content_pooling_layers=[2],
                 optimizer='adam',
                 log_summary=False,
                 preview_before=False,
                 loss_after=100,
                 session=None):
        """Creates a Neural Style Transfer Object that can transfer the style of style image to a content

        Arguments:
            content_pth {str} -- Path of the Content Image
            style_pth {str} -- Path of the Style Image

        Keyword Arguments:
            gen_file_name {str} -- Name of the file styled file (default: {'./styled_image.jpg'})
            epochs {int} -- Number of epochs to build the image (default: {10000})
            initialize_randomly {bool} -- Should we start with noise and generate image or modify content image directly. 
                                            Former will be slow but expect better results (default: {False})
            learning_rate {float} -- Learning Rate (default: {0.5})
            content_factor {float} -- Factor that determines how close the output should look to Content  (default: {0.001})
            style_factor {int} -- Factor that determines how close the output should look to Content (default: {1})
            style_pooling_layers {list} -- Pooling Layer to use for style capture from vgg19 (default: {[3, 4]})
            style_layers_weight {list} -- Weights for each pooling layer in style capturing. This must match with len(style_pooling_layers) (default: {[0.5, 0.5]})
            content_pooling_layers {list} -- The Pooling Layer to use for Content Capturing. (default: {[2]})
            optimizer {str} -- Optimizer to use in training. Fallback to GradientDescent incase not known optimizer passed.  (default: {'adam'})
            log_summary {bool} -- should we dump tensorboard logs (default: {False})
            preview_before {bool} -- should we preview after saving image (default: {False})
            loss_after {int} -- prints loss after this much epochs (default: {100})
            session {[type]} -- Any Session of yours to use. (default: {None})
        """

        self.style_pool = style_pooling_layers
        self.content_pool = content_pooling_layers
        self.style_pool_weight = style_layers_weight
        self.alpha = content_factor
        self.beta = style_factor
        self.epochs = epochs
        self.loss_interval = loss_after
        self.preview = preview_before
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.out_file_name = gen_file_name
        self.summary = log_summary
        self.random_init = initialize_randomly
        self.sess = session or tf.Session()

        self.content_img = None
        self.style_img = None
        self.style_img_new = None

        self.generated_image = None
        self.content_cost = None
        self.style_cost = None
        self.total_cost = None

        self.vgg_placeholder1 = None
        self.vgg_placeholder2 = tf.placeholder(
            tf.float32, shape=(None, None, 3))
        self.dualvgg19 = None

        self._decode_imgs(content_pth, style_pth)
        self._style_image_reshaper()
        self._create_random_noise()
        self._build_vgg19_layer()
        self._style_content_cost_builder()
        self._complete_loss()

    def _decode_imgs(self, content_pth, style_pth):
        c_data = mpimg.imread(content_pth)
        s_data = mpimg.imread(style_pth)
        self.content_img = np.asarray(c_data, np.float32)  # pylint : disable=E1101
        self.style_img = np.asarray(s_data, np.float32)  # pylint : disable=E1101

    def _style_image_reshaper(self):
        with tf.name_scope('style_resizer'):
            self.style_img_new = tf.image.resize_image_with_crop_or_pad(self.vgg_placeholder2,
                                                                        target_height=np.shape(
                                                                            self.content_img)[0],
                                                                        target_width=np.shape(self.content_img)[1])

    def _create_random_noise(self):
        with tf.name_scope('random_noise'):
            if not self.random_init:
                self.generated_image = tf.get_variable(
                    'gen_img', initializer=tf.constant(self.content_img), dtype=tf.float32)
            else:
                self.generated_image = tf.get_variable(
                    'gen_img', shape=np.shape(self.content_img), dtype=tf.float32)

    def _build_vgg19_layer(self):
        with tf.name_scope('vgg19'):
            self.vgg_placeholder1 = tf.placeholder(
                tf.float32, shape=np.shape(self.content_img))
            self.dualvgg19 = ImageNetVGG19(
                self.generated_image, self.vgg_placeholder1, self.style_img_new)

    def _style_content_cost_builder(self):
        res = []
        res2 = []
        i = 0
        with tf.name_scope('vgg19'):
            for layer in [1, 2, 3, 4, 5]:
                with tf.name_scope('vgg19_block_'+str(layer)):
                    c, g, st = tf.unstack(
                        self.dualvgg19['block'+str(layer)+'_pool'])
                    x = tf.reshape(tf.transpose(c, perm=[2, 1, 0]), shape=[
                                   c.shape.as_list()[2], -1])
                    y = tf.reshape(tf.transpose(g, perm=[2, 1, 0]), shape=[
                                   g.shape.as_list()[2], -1])
                    z = tf.reshape(tf.transpose(st, perm=[2, 1, 0]), shape=[
                                   st.shape.as_list()[2], -1])
                    if layer in self.style_pool:
                        style_st = tf.matmul(z, tf.transpose(z))
                        style_y = tf.matmul(y, tf.transpose(y))
                        scal_pix = tf.reduce_prod(
                            tf.square(tf.constant(c.shape.as_list(), dtype=tf.float32)))
                        r = tf.losses.mean_squared_error(
                            style_st, style_y, self.style_pool_weight[i])/(2*scal_pix)
                        res.append(r)
                        i += 1
                    if layer in self.content_pool:
                        res2.append(tf.losses.mean_squared_error(x, y))

            self.style_cost = tf.reduce_sum(tf.stack(res))
            self.content_cost = tf.reduce_sum(tf.stack(res2))

    def _complete_loss(self):
        self.total_cost = (self.alpha * self.content_cost) + \
            (self.beta * self.style_cost)

    def generate(self):
        """starts the model to generate the picture
        """
        optimizer = None
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        elif self.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer
        else:
            print('Optimizer ' + self.optimizer +
                  'is not known. Using Gradient Descent Now.')
            optimizer = tf.train.GradientDescentOptimizer

        train_step = optimizer(self.learning_rate).minimize(
            self.total_cost, var_list=[self.generated_image])

        self.sess.run(tf.global_variables_initializer())

        with self.sess.as_default():
            self.dualvgg19.load_weights()

        raw_gen_out = tf.squeeze(self.generated_image)
        clipped = tf.clip_by_value(
            raw_gen_out, clip_value_min=0, clip_value_max=255)

        if self.summary:
            tf.summary.FileWriter('./logs', tf.get_default_graph())

        feeder = {self.vgg_placeholder1: self.content_img,
                  self.vgg_placeholder2: self.style_img}

        for i in range(self.epochs):
            self.sess.run(train_step, feed_dict=feeder)
            if i % self.loss_interval == 0:
                print(self.sess.run(self.total_cost, feed_dict=feeder))
            # if i % self.preview_interval == 0:
            #   image_out = self.sess.run(clipped)
            #   image_out = np.array(image_out).astype(int)
            #   image_out = np.uint8(image_out)
            #   plt.imshow(image_out)
            #   plt.show()

        if self.preview:
            print('Final Image is : ')
            image_out = self.sess.run(clipped)
            image_out = np.array(image_out).astype(int)
            image_out = np.uint8(image_out)
            plt.imshow(image_out)
            mpimg.imsave(self.out_file_name, image_out)
            self.sess.close()
            plt.show()
        else:
            image_out = self.sess.run(clipped)
            image_out = np.array(image_out).astype(int)
            image_out = np.uint8(image_out)
            mpimg.imsave(self.out_file_name, image_out)
            print('Saved the image as file name', self.out_file_name)
            self.sess.close()
