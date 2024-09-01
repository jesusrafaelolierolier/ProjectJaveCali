import cv2
import os

# Ruta al directorio de imágenes
dataPath = 'C:/Users/JesusOlier/source/repos/Python/ProyectoJave/Fotos/'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Verificación de la disponibilidad de EigenFaceRecognizer
if not hasattr(cv2.face, 'EigenFaceRecognizer_create'):
    print("Error: EigenFaceRecognizer no está disponible en tu instalación de OpenCV.")
    exit()

# Creación del reconocedor de caras EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Carga del modelo entrenado con manejo de excepciones
try:
    face_recognizer.read('modeloEigenFace.xml')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Inicialización de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen desde la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No se detectaron rostros.")
        continue

    for (x, y, w, h) in faces:
        print(f"Rostro detectado en: x={x}, y={y}, w={w}, h={h}")
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (64, 64))  # Cambiar el tamaño a 64x64

        try:
            label, confidence = face_recognizer.predict(rostro)
            accuracy = 100 - (confidence / 100)  # Ajustar la precisión para valores más realistas
            print(f"Predicción: Label={label}, Confianza={confidence}, Accuracy={accuracy}")
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            continue

        if accuracy > 0 and label < len(imagePaths):
            label_text = f'{imagePaths[label]} - {accuracy:.2f}%'
            color = (0, 255, 0)  # Verde para predicciones confiables
        else:
            label_text = 'Desconocido'
            color = (0, 0, 255)  # Rojo para predicciones no confiables

        cv2.putText(frame, label_text, (x, y - 25), 2, 1.1, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
