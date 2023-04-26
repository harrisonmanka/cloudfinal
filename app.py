from flask import Flask, render_template, Response
# from google.cloud import storage
import cv2
# import os
# import numpy as np
# from tensorflow import keras

# authenticate with Google Cloud Storage
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cloud-final-384722-5c828988d2cf.json'
# storage_client = storage.Client.from_service_account_json('./cloud-final-384722-5c828988d2cf.json')
#
# bucket_name = 'cloud-final-fer2013'
# model_path = 'TBD'
# model_file = 'TBD'
# bucket = storage_client.bucket(bucket_name)
# blob = bucket.blob(model_path)
# blob.download_to_filename(model_file)
#
# model = keras.models.load_model(model_file)

app = Flask(__name__)
camera = cv2.VideoCapture(0)


# EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),  # function to generate frames
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == "__main__":
    app.run(debug=True)
