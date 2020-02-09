import tensorflow as tf


class Model(object):
    def __init__(self, img_w, img_h, num_class, num_layers, batch_size, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_class = num_class
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.img_w, self.img_h,3])
        self.target = tf.sparse_placeholder(tf.int32, name='label')
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(inputs= self.inputs,filters=8,kernel_size=3,padding="same",activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
        conv2 = tf.layers.conv2d(inputs= pool1,filters=16,kernel_size=3,padding="same",activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        
        rnn_in = tf.reshape(pool3,[-1,self.img_w//8,self.img_h//8*32])
        
        m_cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for _ in range(self.num_layers)])
        output, _ = tf.nn.dynamic_rnn(m_cell, rnn_in, self.seq_len, dtype=tf.float32, time_major=False)
        h_state = tf.reshape(output, (-1, self.num_units))

        w = tf.Variable(tf.truncated_normal([self.num_units, self.num_class], stddev=0.1))
        b = tf.constant(0.1, dtype=tf.float32, shape=[self.num_class])

        logits = tf.matmul(h_state, w) + b
        logits = tf.reshape(logits, [-1, 67, self.num_class])
        self.logits = tf.transpose(logits, (1, 0, 2))

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated=False)

        self.cost = tf.nn.ctc_loss(labels=self.target, inputs=self.logits, sequence_length=self.seq_len)
        self.loss = tf.reduce_mean(self.cost)

        self.op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)        
        
        self.err = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.target))

    def unit(self):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell
