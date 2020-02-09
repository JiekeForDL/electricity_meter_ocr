import sys
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2

from main import ocr_predict


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.init_button()
        self.init_image_box()
        self.init_text()
        self.show()

    def gen_label(self, text, position, size=(), align=Qt.AlignCenter):
        label = QLabel(self)
        label.setText(text)
        if size:
            label.setFixedSize(size[0], size[1])
        label.move(position[0], position[1])
        if align:
            label.setAlignment(align)
        return label

    def init_text(self):
        self.gen_label('图片名称:', (20, 530), (60, 30))
        self.image_name = self.gen_label('', (80, 530), (450, 30), align=None)
        self.gen_label('第一行:', (20, 560), (60, 30))
        self.line1 = self.gen_label('', (80, 560), (450, 30), align=None)
        self.gen_label('第二行:', (20, 590), (60, 30))
        self.line2 = self.gen_label('', (80, 590), (450, 30), align=None)
        self.gen_label('第三行:', (20, 620), (60, 30))
        self.line3 = self.gen_label('', (80, 620), (450, 30), align=None)
        self.texts = [self.line1, self.line2, self.line3]

    def init_button(self):
        upload_btn = QPushButton(self)
        upload_btn.setText("打开图片")
        upload_btn.move(10, 10)
        upload_btn.clicked.connect(self.openimage)

    def clear_text(self):
        for text in self.texts:
            text.setText('')

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.clear_text()
        self.image_name.setText(imgName.split('/')[-1])
        if imgName:
            jpg = self.resize_image(QPixmap(imgName), self.origin_image)
            self.origin_image.setPixmap(jpg)
            self.ocr_recognoze(imgName)

    def refresh_detect_image(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.detect_image.setPixmap(self.resize_image(QPixmap.fromImage(qImg), self.detect_image))

    def ocr_recognoze(self, image_name):
        image = cv2.imread(image_name)
        predict, detect_img = ocr_predict(image)
        self.refresh_detect_image(detect_img)
        for i, text in enumerate(predict):
            getattr(self, 'line%s' % (i+1)).setText(text)

    def resize_image(self, image, box):
        if image.width() / image.height() >= box.width() / box.height():
            width = box.width()
            height = image.height() * width / image.width()
        else:
            height = box.height()
            width = image.width() * height / image.height()
        return image.scaled(width, height)

    def init_image_box(self):
        self.origin_image = self.gen_label('原始图片', (20, 50), (360, 480))
        self.origin_image.setStyleSheet("QLabel{border:1px solid #014F84;}")

        self.detect_image = self.gen_label('检测图片', (400, 50), (360, 480))
        self.detect_image.setStyleSheet("QLabel{border:1px solid #014F84;}")

    def initUI(self):
        self.resize(780, 660)
        self.center()
        self.setWindowTitle('OCR演示界面')
        self.setObjectName("MainWindow")
        self.setStyleSheet("#MainWindow{background:white;}")

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())