from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from keras.models import load_model

app = Flask(__name__)

model = load_model('model_final.h5')

class_dictionary = {0: 'no_car', 1: 'car'}

cap = cv2.VideoCapture('car_test.mp4')

with open('carposition.pkl', 'rb') as f:
    posList = pickle.load(f)

width, height = 130, 65

def checkParkingSpace(img):
    spaceCounter = 0
    imgCrops = []

    for pos in posList:
        x, y = pos
        imgCrop = img[y:y + height, x:x + width]
        imgResize = cv2.resize(imgCrop, (48, 48))
        imgNormalized = imgResize / 255.0
        imgCrops.append(imgNormalized)

    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(posList):
        x, y = pos
        inID = np.argmax(predictions[i])
        label = class_dictionary[inID]

        if label == 'no_car':
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            textColor = (0,0,0)
        else:
            color = (0, 0, 255)
            thickness = 2
            textColor = (255,255,255)

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        font_scale = 0.5
        text_thickness = 1
        
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 6, textY + 2), color, -1)
        cv2.putText(img, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)

    totalSpaces = len(posList)


    return img, spaceCounter, totalSpaces - spaceCounter

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1280, 720))
        img, free_spaces, occupied_spaces = checkParkingSpace(img)
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (1280, 720))
        _, free_spaces, occupied_spaces = checkParkingSpace(img)
        return jsonify(free=free_spaces, occupied=occupied_spaces)
    return jsonify(free=0, occupied=0)

if __name__ == "__main__":
    app.run(debug=True)
