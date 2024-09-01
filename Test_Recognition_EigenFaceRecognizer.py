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
            image = cv2.imread(filePath, 0)
            if image is not None:
                image = cv2.resize(image, (64, 64))  # Cambiar el tamaño a 64x64
                facesData.append(image)
                labels.append(label)
        label += 1
    else:
        # Si la entrada es un archivo (posiblemente una imagen directamente en el directorio)
        print(f'Archivo de imagen encontrado: {entryPath}')
        label_from_filename = int(entry.partition('_')[0])
        label_dict[label_from_filename] = label_from_filename  # Aquí corregí el nombre de la variable
        print(f'Asignando la etiqueta {label_from_filename} al archivo {entry}')
        image = cv2.imread(entryPath, 0)
        if image is not None:
            image = cv2.resize(image, (64, 64))  # Cambiar el tamaño a 64x64
            facesData.append(image)
            labels.append(label_from_filename)

# Verificar si hay datos suficientes para entrenar
if len(facesData) < 2 or len(set(labels)) < 2:
    print("Error: Necesitas al menos dos imágenes y dos etiquetas diferentes para entrenar el modelo.")
    exit()

print('Etiquetas asignadas:', label_dict)

# Crear el reconocedor de caras EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Entrenar el reconocedor
print("Entrenando - EigenFaceRecognizer -")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.write('modeloEigenFace.xml')
print("Modelo almacenado...")
