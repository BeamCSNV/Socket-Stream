from flask import Flask, render_template, Response
import cv2
import zmq
import pickle
import numpy as np

app = Flask(__name__)

context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://127.0.0.1:5555")

def generate_frames():
    while True:
        frame_data = receiver.recv()
        frame = cv2.imdecode(np.frombuffer(pickle.loads(frame_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
