import cv2
import os

# Ruta al directorio de imágenes
dataPath = 'C:/Users/JesusOlier/source/repos/Python/ProyectoJave/Fotos/'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Verificación de la disponibilidad de FisherFaceRecognizer
if not hasattr(cv2.face, 'FisherFaceRecognizer_create'):
    print("Error: FisherFaceRecognizer no está disponible en tu instalación de OpenCV.")
    exit()

# Creación del reconocedor de caras FisherFace
face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Carga del modelo entrenado con manejo de excepciones
try:
    face_recognizer.read('modeloFisherFace.xml')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Inicialización de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Establecer resolución de la cámara (opcional, para mejorar rendimiento)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Clasificador para detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen desde la cámara.")
        break

    # Conversión de la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # Detección de rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No se detectaron rostros.")
        continue

    for (x, y, w, h) in faces:
        print(f"Rostro detectado en: x={x}, y={y}, w={w}, h={h}")
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (64, 64), interpolation=cv2.INTER_CUBIC)

        try:
            # Predicción del rostro usando el modelo FisherFace
            label, confidence = face_recognizer.predict(rostro)
            accuracy = 100 - confidence
            print(f"Predicción: Label={label}, Confianza={confidence}, Accuracy={accuracy}")
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            continue

        # Mostrar el nombre del archivo y la precisión solo si es aceptable
        if accuracy >= 0 and label < len(imagePaths):
            label_text = f'{imagePaths[label]} - {accuracy:.2f}%'
            color = (0, 255, 0)  # Verde para predicciones confiables
        else:
            label_text = 'Desconocido'
            color = (0, 0, 255)  # Rojo para predicciones no confiables

        # Dibujar el rectángulo y el texto en la imagen
        cv2.putText(frame, label_text, (x, y - 25), 2, 1.1, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Mostrar el fotograma en la ventana
    cv2.imshow('frame', frame)

    # Presionar 'ESC' para salir
    k = cv2.waitKey(1)
    if k == 27:
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
