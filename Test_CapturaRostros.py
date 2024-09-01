import cv2
import os
import imutils
from tkinter import simpledialog

dataPath = 'C:/Users/JesusOlier/source/repos/Python/ProyectoJave/Fotos/'
if not os.path.exists(dataPath):
    print('Carpeta creada: ', dataPath)
    os.makedirs(dataPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
Identificacion = simpledialog.askstring("Identificacion del Empleado", "Digite la Identificación del empleado a capturar el rostro (Deje en BLANCO para CANCELAR): ")

while len(Identificacion) > 0:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        print(f'{count} / 100')  # Cambia el número máximo de imágenes
        cv2.imwrite(f'{dataPath}/{Identificacion}_{count}.jpg', rostro)
        count += 1
    
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
    if count >= 100:  # Cambiado a 100 imágenes
        Identificacion = simpledialog.askstring("Identificacion del Empleado", "Digite la Identificación del empleado a capturar el rostro (Deje en BLANCO para CANCELAR): ")
        count = 0

cap.release()
cv2.destroyAllWindows()
