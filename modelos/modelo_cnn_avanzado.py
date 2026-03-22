import tensorflow as tf
from tensorflow.keras import layers, models

def obtener_modelo_avanzado(input_shape, num_classes):
    model = models.Sequential()
    
    # --- 1. CAPA DE ENTRADA Y DATA AUGMENTATION ---
    # Esto rota, gira y hace zoom a las imágenes en cada época para que la red no memorice.
    # Solo se aplica automáticamente durante el entrenamiento (Train).
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # --- 2. BLOQUES CONVOLUCIONALES ---
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- 3. CLASIFICADOR OPTIMIZADO ---
    # GlobalAveragePooling extrae lo más importante y reduce drásticamente los parámetros
    model.add(layers.GlobalAveragePooling2D()) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model