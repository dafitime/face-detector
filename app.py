from flask import Flask, request, jsonify, render_template
from detect_face import detect_faces
import base64
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_data = base64.b64decode(data['image'].split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite("temp.jpg", img)

    boxes = detect_faces("temp.jpg")
    print("Detected boxes:", boxes)
    return jsonify({"boxes": boxes})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
