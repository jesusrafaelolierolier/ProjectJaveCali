import cv2
import os
import numpy as np

# Ruta al directorio de imágenes
dataPath = 'C:/Users/JesusOlier/source/repos/Python/ProyectoJave/Fotos/'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

labels = []
facesData = []
label_dict = {}

# Preprocesamiento y lectura de imágenes
print('Leyendo las imágenes y asignando etiquetas...')
for entry in imagePaths:
    filePath = os.path.join(dataPath, entry)
    label_from_filename = int(entry.partition('_')[0])

    # Leer la imagen en escala de grises
    image = cv2.imread(filePath, 0)
    if image is None:
        continue

    # Preprocesamiento: Redimensionar y normalizar
    image = cv2.resize(image, (64, 64))
    image = cv2.equalizeHist(image)

    facesData.append(image)
    labels.append(label_from_filename)

    if label_from_filename not in label_dict:
        label_dict[label_from_filename] = entry.partition('_')[2]

print('Etiquetas asignadas:', label_dict)

# Crear el reconocedor de caras EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Entrenar el reconocedor
print("Entrenando - EigenFaceRecognizer -")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.write('modeloEigenFace.xml') 
print("Modelo almacenado...")
