import cv2
import os
import numpy as np

# Ruta al directorio de imágenes
dataPath = 'C:/Users/JesusOlier/source/repos/Python/ProyectoJave/Fotos/'

# Obtener todas las entradas en el directorio
entries = os.listdir(dataPath)
print('Entradas en el directorio:', entries)

labels = []
facesData = []
label = 0
label_dict = {}

print('Leyendo las imágenes y asignando etiquetas...')

# Procesar cada entrada en el directorio
for entry in entries:
    entryPath = os.path.join(dataPath, entry)
    if os.path.isdir(entryPath):
        # Si la entrada es un directorio, se procesa como una persona con múltiples imágenes
        label_dict[label] = entry  # Mapear la etiqueta al nombre
        for fileName in os.listdir(entryPath):
            filePath = os.path.join(entryPath, fileName)
            print('Rostro:', filePath)
            # Leer la imagen en escala de grises
            image = cv2.imread(filePath, 0)
            # Redimensionar la imagen
            image = cv2.resize(image, (150, 150))
            # Añadir la imagen y la etiqueta correspondiente a las listas
            facesData.append(image)
            labels.append(label)
        label += 1
    else:
        # Si la entrada es un archivo (posiblemente una imagen directamente en el directorio)
        print('Archivo de imagen encontrado:', entryPath)
        # Asume que el nombre del archivo incluye la etiqueta
        label_from_filename = int(entry.partition('_')[0])
        print(f'Asignando la etiqueta {label_from_filename} al archivo {entry}')
        # Leer la imagen en escala de grises
        image = cv2.imread(entryPath, 0)
        # Redimensionar la imagen
        image = cv2.resize(image, (150, 150))
        # Añadir la imagen y la etiqueta correspondiente a las listas
        facesData.append(image)
        labels.append(label_from_filename)

print('labels=', labels)
print('Etiquetas asignadas:', label_dict)

# Crear el reconocedor de caras LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
#face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

# Entrenar el reconocedor
print("Entrenando - LBPHFaceRecognizer -")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.write('modeloLBPHFace.xml') 
print("Modelo almacenado...")
