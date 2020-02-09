import os
import numpy as np
import cv2
import tensorflow as tf

from tf_crnn.tf_crnn_model import Model
from tf_crnn.utils import TfCrnnModelData, reshape_image
from tf_crnn.settings import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_layers = 2
seq_len = 67
num_units = 256


class CRNN(object):

    def __init__(self):
        self.data = TfCrnnModelData()
        self.model = Model(img_w=image_width,
                           img_h=image_height,
                           num_class=num_output,
                           batch_size=1,
                           num_units=num_units,
                           num_layers=num_layers)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, 'tf_crnn/model/model.ckpt-53999')  # 43999
        print('Load crnn model successfully')

    def predict(self, img_list):
        inputs = []
        for image in img_list:
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])
            inputs.append(reshape_image(image))
        inputs = np.asarray(inputs)
        inputs = np.swapaxes(inputs, 1, 2)

        batch_size = inputs.shape[0]
        seq_lens = np.ones(batch_size) * (seq_len)

        feed = {
            self.model.inputs : inputs,
            self.model.seq_len : seq_lens,
            self.model.keep_prob : 1.0
        }

        decode = self.sess.run(self.model.decoded, feed_dict=feed)
        pre = self.data.decode_sparse_tensor(decode[0])
        # print('predict: ', pre)
        return pre
