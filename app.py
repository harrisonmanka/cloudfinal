from flask import Flask, render_template, Response
from google.cloud import storage
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
camera = cv2.VideoCapture(0)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model_path = "D:/CS493/finalproject/cloudfinal/face_model.h5"
model = load_model(model_path)

cascade = 'D:/CS493/finalproject/cloudfinal/haarcascade_frontalface_default.txt'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),  # function to generate frames
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


faceCascade = cv2.CascadeClassifier(cascade)


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            all_faces = faceCascade.detectMultiScale(roi_gray)
            if len(all_faces) == 0:
                print("FACE NOT DETECTED")
            else:
                for (ex, ey, ew, eh) in all_faces:
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            predictions = model.predict(final_image)
            max_index = int(np.argmax(predictions))
            predicted_emotion = emotions[max_index]
            font_scale = 1.5
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, predicted_emotion, (x, y), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == "__main__":
    app.run(debug=True)
