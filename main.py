import cv2
import numpy as np

from keras_yolo3.yolo import YOLO
from tf_crnn.tf_predict import CRNN


crnn = CRNN()
yolo3 = YOLO()


def ocr_predict(image):
    text1 = []
    text3 = []
    detect_img = image.copy()
    for label, score, img, box in yolo3.get_text_box_area(image):
        cv2.rectangle(detect_img, box[0], box[1], (0, 0, 255), thickness=2)
        if label == 'text1':
            text1.append((score, img))
        elif label == 'text3':
            text3.append((score, img))
    text1 = sorted(text1, key=lambda x: -x[0])[:1]
    text3 = sorted(text3, key=lambda x: -x[0])[:2]
    predict = crnn.predict(np.asarray(text1 + text3)[:, 1])
    return predict, detect_img


if __name__ == '__main__':
    while True:
        command = input('Please input image path: ')
        if command == 'quit':
            break
        image = cv2.imread(command)
        pre, image = ocr_predict(image)
        print('predict: ', pre)