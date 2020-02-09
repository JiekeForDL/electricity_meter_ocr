import os
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    yolo = YOLO()
    for image_name in os.listdir('data/images/'):
        image = cv2.imread('data/images/' + image_name)
        for label, img in yolo.get_text_box_area(image):
            # print(label)
            # plt.imshow(img)
            # plt.show()
            plt.imsave('data/text_box/%s_%s.jpg' % (image_name.split('.')[0], label), img)