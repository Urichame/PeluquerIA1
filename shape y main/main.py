from flask import Flask, jsonify
import cv2
import dlib

app = Flask(__name__)

# Inicializar detector y predictor de rostros
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

# Diccionario de recomendaciones de cortes
recomendaciones_cortes = {
    "Cara ovalada": "French crop y taper fade",
    "Cara redonda": "Hugh fade y burst fade con texturizado",
    "Cara cuadrada": "Low fade y mid fade con el pelo largo",
    "Cara corazón": "Skin fade y buzz cut",
    "Cara diamante": "French crop y mid fade con texturizado"
}

@app.route('/run-script', methods=['GET'])
def run_script():
    # Iniciar la cámara
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"message": "Error al capturar video"})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detector(gray)
        for rostro in rostros:
            landmarks = predictor(gray, rostro)
            puntos = landmarks.parts()
            forma_cara = determinar_forma_del_rostro(puntos)
            corte_recomendado = recomendaciones_cortes.get(forma_cara, "No encontrado")

            # Dibujar puntos y mostrar información
            for p in puntos:
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)
            cv2.putText(frame, forma_cara, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, corte_recomendado, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Cámara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Detección completada"})

if __name__ == "__main__":
    app.run(debug=True)