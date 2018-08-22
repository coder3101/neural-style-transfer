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

import os.path
import warnings

import numpy as np
import tensorflow as tf

warnings.simplefilter('ignore', category=FutureWarning)


class ImageNetVGG19:
    def __init__(self,
                 gen_img,
                 input_tensor_placeholder1,
                 input_tensor_placeholder2,
                 sess=None,
                 name=None):
        """Creates the graph of VGG-19 and loads the weights of Pre-trained Imagenet

        Arguments:
            gen_img {tf.Variable} -- The target Image to train On 
            input_tensor_placeholder1 {tf.Placeholder} -- The Content Image Placeholder
            input_tensor_placeholder2 {tf.Placeholder} -- The Style Image Placeholder

        Keyword Arguments:
            sess {[type]} -- Explicit Session to run this Model on. None creates a new Session (default: {None})
            name {[type]} -- Use to dumps weights to the named Folder (default: {None})
        """

        self.input_tensor = input_tensor_placeholder1
        self.input_tensor_style = input_tensor_placeholder2
        self.session = sess or tf.Session()
        self.gen_img = gen_img
        self.name = name or './saved_weights'
        self.vgg19 = None
        self.layers = None
        self.feed_in = None
        self.weights = None
        self.checkpoint = None
        self._preprocess()
        self._build_graph()
        self._save_and_write_weights()

    def _preprocess(self):
        with tf.name_scope('preprocess'):
            RGB_MEAN = np.array([123.68, 116.779, 103.939]).reshape(
                (1, 1, 1, 3)).astype(np.float32)  # pylint: disable=E1101
            img1 = self.input_tensor - RGB_MEAN
            self.input_tensor = tf.squeeze(tf.reverse(img1, axis=[-1]))
            img2 = self.gen_img - RGB_MEAN
            self.gen_img = tf.squeeze(tf.reverse(img2, axis=[-1]))
            img3 = self.input_tensor_style - RGB_MEAN
            self.input_tensor_style = tf.squeeze(tf.reverse(img3, axis=[-1]))

        with tf.name_scope('stacker'):
            self.feed_in = tf.stack(
                [self.input_tensor, self.gen_img, self.input_tensor_style])

    def _build_graph(self):
        with self.session.as_default():  # pylint: disable=E1129
            with tf.variable_scope('vgg_model'):
                self.vgg19 = tf.keras.applications.VGG19(include_top=False,
                                                         input_tensor=self.feed_in,
                                                         weights='imagenet')
            self.layers = {
                layer.name: layer.output for layer in self.vgg19.layers}

            self.weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg19/vgg_model')

    def _save_and_write_weights(self):
        if not os.path.exists(self.name):
            os.mkdir(self.name)

        self.checkpoint = tf.train.Saver(self.weights).save(
            self.session, self.name + '/vgg-19-saved')

        tf.train.write_graph(tf.get_default_graph(),
                             logdir=self.name, name='vgg19.pbtxt')
        self.session.close()

    def load_weights(self):
        """restores all the weights from the saved checkpoints after 
        tf.global_variable_initializer().run() destroys the imagenet weights.
        It must be called before Using this wrapper and must be called within the same
        tf.Session() as training Session().
        """

        tf.train.Saver(self.weights).restore(
            tf.get_default_session(), self.checkpoint)

    def __getitem__(self, key):
        """returns a operation in the vgg by the name

        Arguments:
            key {str} -- the name of op

        Returns:
            tf.Tensor -- Tensor
        """

        return self.layers[key]
