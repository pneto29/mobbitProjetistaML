import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import os
import matplotlib.pyplot as plt

# Carrega metadados de carros a partir de um arquivo .mat e extrai informações de classes e nomes de arquivo.
data = scipy.io.loadmat('bmw10_annos.mat')
annos = data['annos']
classes = annos['class'][0]
filenames = annos['fname'][0]

# Mapeia classes específicas para serem consideradas no modelo, descartando outras.
mapped_classes = np.where(np.isin(classes, [3, 4, 5]), classes, 0)
image_class_mapping = {fname[0]: int(cls) for fname, cls in zip(filenames, mapped_classes)}

# Divide as imagens em conjuntos de treinamento e teste com uma distribuição estratificada.
file_paths = list(image_class_mapping.keys())
labels = list(image_class_mapping.values())
paths_train, paths_test, labels_train, labels_test = train_test_split(
    file_paths, labels, test_size=0.30, stratify=labels, random_state=42)

# Função para organizar imagens em diretórios adequados para treinamento e teste.
def prepare_directories(paths, labels, base_dir):
    for path, label in zip(paths, labels):
        dest_dir = os.path.join(base_dir, str(label))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        src_path = os.path.join('bmw10_ims', path)
        dest_path = os.path.join(dest_dir, path.split('/')[-1])
        copyfile(src_path, dest_path)

prepare_directories(paths_train, labels_train, 'cars_train')
prepare_directories(paths_test, labels_test, 'cars_test')

# Definindo o conjunto de hiperparâmetros para teste
learning_rates = [0.001, 0.0005, 0.0001]
dropout_rates = [0.3, 0.5, 0.7]

best_accuracy = 0
best_params = {}
best_history = None
best_model = None

# Configura geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'cars_train', target_size=(150, 150), batch_size=20, class_mode='sparse')
test_generator = test_datagen.flow_from_directory(
    'cars_test', target_size=(150, 150), batch_size=20, class_mode='sparse')

for lr in learning_rates:
    for dropout in dropout_rates:
        # Define e compila o modelo
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(dropout),
            Dense(5, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Treina o modelo usando os geradores de dados, validando com o conjunto de teste.
        history = model.fit(train_generator, epochs=10, validation_data=test_generator, verbose=0)

        # Avaliar o modelo treinado.
        accuracy = max(history.history['val_accuracy'])

        # Atualizar os melhores parâmetros, se necessário.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'learning_rate': lr, 'dropout_rate': dropout}
            best_history = history  # Salva o histórico do melhor modelo
            best_model = model

# Avaliação detalhada do melhor modelo
test_labels = test_generator.classes
predictions = best_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Gerando o relatório de classificação
report = classification_report(test_labels, predicted_classes, target_names=test_generator.class_indices.keys())
print(report)

# Exibindo o relatório e melhores parâmetros
print(f"Melhores parâmetros: {best_params}")
print(f"Melhor acurácia de validação: {best_accuracy}")

# Plotar a evolução do loss e da acurácia durante as épocas de treinamento e validação para o melhor modelo.
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Loss de Treinamento')
plt.plot(best_history.history['val_loss'], label='Loss de Validação')
plt.title('Evolução do Loss por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(best_history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Evolução da Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.show()
