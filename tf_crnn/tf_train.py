import os
import cv2
import numpy as np
import tensorflow as tf
from tf_crnn.tf_crnn_model import Model

from tf_crnn.utils import TfCrnnModelData
from tf_crnn.settings import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_layers = 2
seq_len = 67
num_units = 256
step = 100000
# learn_rate = 0.001

model_dir = 'model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model = Model(img_w=image_width,
              img_h=image_height,
              num_class=num_output,
              batch_size=batch_size,
              num_units=num_units,
              num_layers=num_layers)
data = TfCrnnModelData()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    seq_lens = np.ones(batch_size) * (seq_len)
    acc = 0
    test_data, text_label = data.all_data()
    print('train start')
    for i in range(step):
        inputs, labels = data.next_batch()
        feed = {
            model.inputs : inputs,
            model.target : labels,
            model.seq_len : seq_lens,
            model.keep_prob : 0.5,
            model.lr : learning_rate
        }
        sess.run(model.op, feed_dict=feed)

        if (i+1) % 100 == 0 or i == step-1:
            feed = {
                model.inputs: test_data,
                model.target: text_label,
                model.seq_len: np.ones(len(test_data)) * (seq_len),
                model.keep_prob: 1,
                model.lr: learning_rate
            }
            loss, err, decode = sess.run([model.loss, model.err, model.decoded], feed_dict=feed)
            ori = data.decode_sparse_tensor(text_label)
            pre = data.decode_sparse_tensor(decode[0])
            acc = data.hit(pre, ori)
            msg = 'train step: %d, accuracy: %.4f, word error: %.6f, loss: %f, lr: %f' % (i+1, acc, err, loss, learning_rate)
            print(msg)
            print('ori: %s\npre: %s' % (ori[0], pre[0]))

        if acc >= 0.95 and ((i+1) % 1000 == 0 or i == step-1):
            checkpoint_path = os.path.join(model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)
            print('save model: ' + checkpoint_path)
            learning_rate = max(0.000001, learning_rate * 0.98)