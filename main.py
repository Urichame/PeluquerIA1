from flask import Flask, Response
import cv2
import dlib

app = Flask(__name__)

# Cargar el detector y predictor de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Función para determinar la forma del rostro
def determinar_forma_del_rostro(puntos):
    ancho_frente = puntos[16].x - puntos[0].x
    alto_frente = puntos[8].y - puntos[19].y
    alto_mandibula = puntos[8].y - puntos[27].y
    ancho_mandibula = puntos[16].x - puntos[0].x

    if ancho_frente / alto_frente > 1.5:
        return "Cara cuadrada"
    elif alto_frente / ancho_frente > 1.3 and alto_mandibula / ancho_mandibula < 0.75:
        return "Cara ovalada"
    elif alto_frente / ancho_frente < 1.2 and ancho_mandibula / alto_mandibula > 1.2:
        return "Cara diamante"
    elif puntos[27].x - puntos[0].x < puntos[8].y - puntos[27].y:
        return "Cara corazón"
    else:
        return "Cara redonda"

# Recomendaciones según la forma de la cara
recomendaciones_cortes = {
    "Cara ovalada": "French crop y taper fade",
    "Cara redonda": "Hugh fade y burst fade con texturizado",
    "Cara cuadrada": "Low fade y mid fade con el pelo largo",
    "Cara corazón": "Skin fade y buzz cut",
    "Cara diamante": "French crop y mid fade con texturizado"
}

# Ruta que captura el video
@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostros = detector(gray)

            for rostro in rostros:
                landmarks = predictor(gray, rostro)
                puntos = [(p.x, p.y) for p in landmarks.parts()]
                forma_cara = determinar_forma_del_rostro(landmarks.parts())
                corte_recomendado = recomendaciones_cortes.get(forma_cara, "No encontrado")

                for p in puntos:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)

                cv2.putText(frame, forma_cara, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, corte_recomendado, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
