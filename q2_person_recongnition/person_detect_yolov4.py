"""
Autor: Polycarpo Neto
Data: 17 de Maio de 2024
Descrição: Este script carrega uma imagem e realiza contagem de pessoas por meio de yolov4.
"""
import cv2
import numpy as np

# Carregar a rede YOLOv4
weights_path = 'yolov4.weights'  # Caminho para o arquivo de pesos do modelo YOLOv4
config_path = 'yolov4.cfg'       # Caminho para o arquivo de configuração do modelo YOLOv4
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Definir o backend de computação como OpenCV
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # Definir o dispositivo alvo como CPU

# Carregar a imagem a partir de um arquivo
image_path = 'image02.png'                            # Caminho para a imagem a ser processada
image = cv2.imread(image_path)                        # Ler a imagem usando OpenCV
height, width = image.shape[:2]                       # Obter as dimensões da imagem

# Preparar a entrada para a rede
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)  # Converter a imagem para o formato compatível com o YOLO
net.setInput(blob)                                    # Definir o blob como entrada da rede

# Executar a detecção
layer_names = net.getLayerNames()                     # Obter os nomes de todas as camadas da rede
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]  # Identificar as camadas de saída onde queremos extrair as informações
outs = net.forward(output_layers)                     # Realizar a inferência e obter as detecções das camadas de saída

# Processar as detecções
conf_threshold = 0.5                                  # Limiar de confiança para filtrar detecções fracas
nms_threshold = 0.4                                   # Limiar para Non-Maximum Suppression (NMS)
class_ids = []                                        # Lista para armazenar IDs de classes das detecções
confidences = []                                      # Lista para armazenar confianças das detecções
boxes = []                                            # Lista para armazenar caixas delimitadoras das detecções

# Loop através de cada detecção nas camadas de saída
for out in outs:
    for detection in out:
        scores = detection[5:]                        # Extrair a pontuação de confiança para todas as classes
        class_id = np.argmax(scores)                   # Obter o índice da classe com a maior pontuação
        confidence = scores[class_id]                  # Obter a pontuação de confiança para a classe com maior pontuação

        if confidence > conf_threshold and class_id == 0:  # Filtrar detecções por limiar de confiança e classe 'pessoa'
            # Escalar as coordenadas da caixa delimitadora de volta ao tamanho original da imagem
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calcular as coordenadas do canto superior esquerdo da caixa
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Adicionar caixa delimitadora à lista de caixas
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar Non-Maximum Suppression para reduzir o número de caixas delimitadoras sobrepostas
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
count_people = len(indices)  # Contar o número de pessoas detectadas

print(f"Number of people detected: {count_people}")

# Plotar retângulos e números nas pessoas detectadas
for index, i in enumerate(indices.flatten()):
    box = boxes[i]
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenhar o retângulo
    cv2.putText(image, str(index + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Numerar a pessoa

# Salvar a imagem com as detecções
cv2.imwrite('Yolov4_person.png', image)

# Opcional: Mostrar a imagem com as detecções na tela
cv2.imshow("Detected People", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
