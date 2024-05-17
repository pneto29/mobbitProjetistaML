"""
Autor: Polycarpo Neto
Data: 17 de Maio de 2024
Descrição: Este script carrega uma imagem e realiza um "count" de estruturas na imagem.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem do caminho especificado em escala de cinza
img_path = 'image01.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Aplica uma binarização na imagem original
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Encontra os contornos na imagem original
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Imagem melhorada aplicando equalização de histograma
img_enhanced = cv2.equalizeHist(img)

# Aplica uma binarização na imagem melhorada
_, thresh_enhanced = cv2.threshold(img_enhanced, 128, 255, cv2.THRESH_BINARY)

# Encontra os contornos na imagem melhorada
contours_enhanced, _ = cv2.findContours(thresh_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Conta o número de contornos encontrados nas duas imagens
object_count = len(contours)
object_count_enhanced = len(contours_enhanced)

# Converte as imagens de escala de cinza para BGR para permitir desenhar contornos coloridos
img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_enhanced_colored = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)

# Itera sobre cada contorno encontrado, desenha-o e numera na imagem original
for i, contour in enumerate(contours):
    cv2.drawContours(img_colored, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img_colored, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Itera sobre cada contorno encontrado, desenha-o e numera na imagem melhorada
for i, contour in enumerate(contours_enhanced):
    cv2.drawContours(img_enhanced_colored, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img_enhanced_colored, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Plota as três imagens lado a lado
plt.figure(figsize=(15, 5))

# Plota a imagem original em escala de cinza
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Plota a imagem com objetos numerados na imagem original
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
plt.title('Original com Objetos Numerados')
plt.axis('off')

# Exibe a figura com as três imagens
plt.show()

# Imprime o número de objetos detectados
print("Número de objetos na imagem original:", object_count)
#print("Número de objetos na imagem melhorada:", object_count_enhanced)

