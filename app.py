from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data_image):
    # Decodifica la imagen de base64 a una imagen OpenCV
    image_data = base64.b64decode(data_image.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Procesa la imagen aquí (por ejemplo, detectar rostros o recomendar cortes)

    # Puedes devolver la respuesta al cliente
    emit('response', {'result': 'Procesado exitosamente!'})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000)
