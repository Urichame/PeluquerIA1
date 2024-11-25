from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import threading
import cv2
import dlib
from fastapi import FastAPI
import uvicorn
from typing import Union
import google.generativeai as genai
import requests
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, File, UploadFile
import os
import requests
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

app = FastAPI()

# Montar una carpeta "static" donde estará Pagina7.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Verificación de carga del predictor
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("¡Predictor cargado correctamente!")
except Exception as e:
    print(f"Error al cargar el predictor: {e}")

# Configurar el detector de rostros
detector = dlib.get_frontal_face_detector()

# Función principal de reconocimiento facial
def run_camera():
    recomendaciones_cortes = {
        "Cara ovalada": "French crop y taper fade",
        "Cara redonda": "Hugh fade y burst fade con texturizado",
        "Cara cuadrada": "Low fade y mid fade con el pelo largo",
        "Cara corazón": "Skin fade y buzz cut",
        "Cara diamante": "French crop y mid fade con texturizado"
    }

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

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detector(gray)
        for rostro in rostros:
            landmarks = predictor(gray, rostro)  # Aquí se usa el predictor cargado anteriormente
            puntos = [(p.x, p.y) for p in landmarks.parts()]

            forma_cara = determinar_forma_del_rostro(landmarks.parts())
            corte_recomendado = recomendaciones_cortes.get(forma_cara, "No encontrado")

            for p in puntos:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            cv2.putText(frame, forma_cara, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, corte_recomendado, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Cámara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ruta para ejecutar la cámara
@app.post("/start-camera")
def start_camera():
    thread = threading.Thread(target=run_camera)
    thread.start()
    return {"message": "Cámara iniciada"}

@app.post("/ia")
async def llamarIa():
    # Llamas a la IA
    return {"message": "IA iniciada"}

# Ruta para redirigir a la página Pagina7.html
@app.get("/redirect-to-page7")
def redirect_to_page7():
    return RedirectResponse(url="/static/Pagina7.html")

# Ruta para servir la página principal
@app.get("/", response_class=HTMLResponse)
def serve_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Peluquería</title>
    </head>
    <body>
        <h1>Debes aparecer así en la cámara</h1>
        <button onclick="startCamera()">Abrir Cámara</button>

        <script>
            function startCamera() {
                fetch('/start-camera', { method: 'POST' })
                    .then(() => {
                        setTimeout(() => {
                            window.location.href = '/static/Pagina7.html';
                        }, 5000); // Espera 5 segundos antes de redirigir
                    })
                    .catch(error => console.error('Error:', error));
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)