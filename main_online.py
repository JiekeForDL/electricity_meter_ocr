import numpy as np
from datetime import datetime
import cv2
from flask import Flask, render_template, request

from main import ocr_predict
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def get_ocr_page():
    if request.method == 'GET':
        return render_template('ocr.html')
    else:
        image = request.files['file'].read()
        fname = datetime.now().strftime('%y%m%d%H%M%S')
        with open('static/raw_images/%s.jpg' % fname, 'wb') as f:
            f.write(image)
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        predict_texts, detect_img = ocr_predict(image)
        cv2.imwrite('static/detect_images/%s.jpg' % fname, detect_img)
        return render_template('ocr.html', predict_texts=predict_texts, raw_img='raw_images/%s.jpg' % fname,
                               detect_img='detect_images/%s.jpg' % fname)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False)