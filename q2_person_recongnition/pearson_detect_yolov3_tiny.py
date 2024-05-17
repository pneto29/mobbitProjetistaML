"""
Autor: Polycarpo Neto
Data: 17 de Maio de 2024
Descrição: Este script carrega uma imagem e realiza contagem de pessoas por meio de yolov3-tiny.
"""

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Baixa o arquivo de pesos da rede YOLOv3-tiny se não estiver presente localmente.
weights_path = 'yolov3-tiny.weights'
config_path = 'yolov3-tiny.cfg'
url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
if not os.path.exists(weights_path):
    response = requests.get(url)
    with open(weights_path, 'wb') as f:
        f.write(response.content)

# Carrega a configuração e os pesos para inicializar a rede neural.
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Carrega uma imagem do disco.
img_path = 'image02.png'
image = cv2.imread(img_path)
image_people = image.copy()  # Para detecção de pessoas.
image_faces = image.copy()   # Para detecção de rostos.

# Obtém dimensões da imagem.
height, width = image.shape[:2]

# Prepara o blob da imagem e realiza a passagem para frente (forward pass).
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

# Inicializa listas para armazenar informações dos objetos detectados.
boxes = []
confidences = []
class_ids = []

# Processa cada saída da rede.
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3 and class_id == 0:
            box = detection[0:4] * np.array([width, height, width, height])
            centerX, centerY, w, h = box.astype('int')
            x = int(centerX - w / 2)
            y = int(centerY - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

for i in indices.flatten():
    x, y, w, h = boxes[i]
    cv2.rectangle(image_people, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Detecção de rostos usando Haar Cascade.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image_faces, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(image_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Plota as imagens.
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_people, cv2.COLOR_BGR2RGB))
plt.title('Detecção de Pessoas')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_faces, cv2.COLOR_BGR2RGB))
plt.title('Detecção de Rostos')
plt.axis('off')

plt.show()
