import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from tf_crnn.settings import *


def check_or_create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def reshape_image(image):
    h, w, c = image.shape
    image = cv2.resize(image, (image_height * w // h, image_height), interpolation=cv2.INTER_AREA)
    img_arr = np.zeros(shape=[image_height, image_width, channels], dtype=np.int32)
    img_arr[:, :image_height * w // h, :] = image
    return img_arr / 255


def get_train_data():
    label_dict = {l: i for i, l in enumerate(labels)}
    img_list = []
    label_list = []
    for img_name in os.listdir(train_path):
        img = cv2.imread(os.path.join(train_path, img_name))
        img_list.append(reshape_image(img))
        label_list.append([label_dict[s] for s in img_name[:-4].split('_')[0]])
    return np.array(img_list), np.array(label_list)


class TfCrnnModelData(object):
    """docstring for Da"""
    def __init__(self):
        self.imgs, self.labels = None, None

    def load_data(self):
        self.imgs, self.labels = get_train_data()

    def next_batch(self):
        if self.imgs is None: self.load_data()
        indics = [random.randrange(0, len(self.imgs)) for _ in range(batch_size)]
        inputs, labels = self.imgs[indics], self.sparse_tuple_from(self.labels[indics])
        inputs = np.swapaxes(inputs, 1, 2)
        return inputs, labels

    def all_data(self):
        if self.imgs is None: self.load_data()
        inputs, labels = self.imgs, self.sparse_tuple_from(self.labels)
        inputs = np.swapaxes(inputs, 1, 2)
        return inputs, labels

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []        
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
     
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)     
        return indices, values, shape

    def decode_sparse_tensor(self, sparse_tensor):
        """Transform sparse to sequences ids."""
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

        result = []
        for index in decoded_indexes:
            ids = [sparse_tensor[1][m] for m in index]
            text = ''.join(list(map(self.id2word, ids)))
            result.append(text)
        return result

    def hit(self, text1, text2):
        """Calculate accuracy of predictive text and target text."""
        res = []
        for idx, words1 in enumerate(text1):
            res.append(words1 == text2[idx])
        return np.mean(np.asarray(res))

    def id2word(self, idx):
        return labels[idx]


if __name__ == '__main__':
    data = TfCrnnModelData()
    img, label = data.next_batch()