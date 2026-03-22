# Proyecto-Aprendizaje-Profundo

# 1. Descripción del Problema

La gestión eficiente de residuos es uno de los mayores desafíos medioambientales actuales. La separación manual es costosa y propensa a errores. Este proyecto busca automatizar la identificación de materiales (vidrio, papel, cartón, plástico, metal y otros) utilizando Redes Neuronales Convolucionales (CNN) para facilitar procesos de clasificación automática en plantas de reciclaje o aplicaciones ciudadanas.

# 2. Conjunto de Datos
Utilizaremos el dataset Garbage Classification, el cual contiene aproximadamente 2,500 imágenes distribuidas en 6 categorías principales: 
- Cartón (Cardboard)
- Vidrio (Glass)
- Metal (Metal)
- Papel (Paper)
- Plástico (Plastic)
- Basura mixta (Trash)

Los datos presentan variaciones en iluminación y escala, lo que supone un reto realista para el entrenamiento de modelos de visión artificial. Los datos pueden descargarse desde este enlace a Kaggle: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification. Deben descomprimirse en la carpeta /data siguiendo la estructura de subcarpetas por clase.

# 3. Métricas de Evaluación
Para medir el éxito de nuestros modelos, utilizaremos:
- Accuracy (Precisión Global): Para evaluar el rendimiento general del sistema.
- Matriz de Confusión: Para identificar qué materiales específicos confunde el modelo (ej. plástico vs. vidrio).
- F1-Score (Macro Average): Para asegurar que el modelo sea robusto en todas las clases, independientemente del número de imágenes por categoría.

# 4. Estructura del Trabajo
Siguiendo la planificación de la asignatura, el proyecto se divide en las siguientes fases:
- Fase 1: Selección y EDA: Definición del problema, carga de datos y análisis exploratorio.
- Fase 2: Modelo Simple: Implementación de una red neuronal básica (MLP o CNN sencilla) como punto de partida.
- Fase 3: Modelo Complejo: Implementación de arquitecturas avanzadas (Transfer Learning, Fine-tuning o arquitecturas personalizadas profundas).
- Fase 4: Informe Final y Presentación: Documentación detallada de los experimentos, resultados obtenidos y defensa oral (Abril).

# 5. Estructura del Repositorio
├── data/               # Instrucciones y scripts para carga de datos
├── notebooks/          # Notebooks de Jupyter
├── README.md           # Descripción del proyecto
└── requirements.txt    # Librerías necesarias 

# 6. Tabla primeros modelos
Vamos a poner en una primera tabla los resultados de unos primeros modelos muy simples:

| Modelo | Parámetros | Train Acc | Val Acc | Test Acc | Notas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística (Lineal)** | ~295,000 | 0.7611 | 0.4407 | **0.4545** | No converge y sobreentrena |
| **Random Forest (ML)** | 5,902 | 0.7729 | 0.6126 | **0.6186** | Limitamos profundidad para evitar sobreentrenamiento |
| **Red Neuronal Simple (MLP)** | 393,446 | 0.2350 | 0.2352 | **0.2352** | Modelo demasiado simple para sacar conclusiones |

---

## Detalles de cada modelo

### 1. Regresión Logística (Modelo Lineal)
Fue el primer intento y el resultado ha sido de un **45.45%** de acierto. El problema principal es que este modelo intenta separar los materiales usando "líneas rectas", pero las imágenes son mucho más complejas que eso por lo que es imposible de resolver. Identifica bien el papel, pero falla mucho con el vidrio y el plástico.

### 2. Random Forest (Machine Learning)
Es el modelo básico que mejor ha funcionado con un **61.86%** de acierto. Tras ajustar la profundidad para que no se limitara a memorizar las fotos (sobreentrenamiento), conseguimos un modelo equilibrado de **5,902 nodos**, muy sólido reconociendo papel y cartón.

### 3. Red Neuronal Simple (MLP)
Curiosamente, este modelo dio el peor resultado (**23.52%**). Las gráficas muestran que el modelo se estanca rápido: tiene muchos parámetros (**393,446**), pero no sabe usarlos para entender el problema, lo que confirma que para imágenes necesitamos capas convolucionales.

## 🧠 Evolución de los Modelos de Deep Learning

Para esta fase del proyecto, se ha abandonado el enfoque de Machine Learning clásico (Random Forest, Regresión, etc.) para implementar Redes Neuronales Convolucionales (CNN) capaces de extraer características espaciales de las imágenes. 

En lugar de entrenar un único modelo, se ha seguido una metodología iterativa para comprender y mitigar los problemas clásicos de las redes profundas con datasets limitados, especialmente el *Overfitting*.

### 1. Fase 1: CNN Básica sin Regularización de Entrada
* **Archivo:** `modelo_cnn_1.py` / Notebook correspondiente (`modelo_cnn_entrega3.ipynb`)
* **Descripción:** Hemos implementado una arquitectura profunda con bloques de Convolución (32, 64 y 128 filtros), Batch Normalization, Global Average Pooling y Dropout en las capas densas. Sin embargo, no hemos aplicado ninguna transformación a las imágenes de entrada.
* **Problema detectado:** *Overfitting extremo*. Al tener pocas imágenes, la red memorizaba el conjunto de entrenamiento (Train Acc cercano al 100%), mientras que el Validation Loss dejaba de mejorar muy pronto, obligando al *Early Stopping* a cortar el entrenamiento tempranamente.
* **Resultados:**
  * Accuracy Train: `0.2337`%
  * Accuracy Val: `0.2372`%
  * Accuracy Test: `0.2352`%

### 2. Fase 2: CNN Avanzada con Data Augmentation
* **Archivo:** `modelo cnn avanzado.ipynb`
* **Descripción:** Para evitar que el modelo memorizase las imágenes, hemos integrado una capa de **Data Augmentation** al inicio de la red (`RandomFlip`, `RandomRotation`, `RandomZoom`). De esta forma, el modelo nunca ve exactamente la misma imagen dos veces por época.
* **Problema detectado:** *Inestabilidad en la convergencia*. Aunque se redujo drásticamente el overfitting, la curva del Validation Loss mostraba grandes rebotes. El optimizador daba pasos demasiado grandes al acercarse a la solución óptima, pasándose de largo constantemente.
* **Resultados:**
  * Accuracy Train: `0.9215`%
  * Accuracy Val: `0.7925`%
  * Accuracy Test: `0.7767`%

### 3. Fase 3: CNN Pro Definitiva (El Modelo Final)
* **Archivo:** `modelo cnn avanzado_final.ipynb`
* **Descripción:** Al modelo anterior se le añadió el callback **`ReduceLROnPlateau`**. Esta técnica monitoriza el Validation Loss y, si detecta un estancamiento durante 5 épocas, reduce el *Learning Rate* a la mitad.
* **Conclusión:** *Convergencia perfecta*. Las gráficas se volvieron completamente suaves y estables. El modelo ha podido seguir aprendiendo durante más de 80 épocas sin rebotar, logrando el mayor nivel de generalización y superando ampliamente la barrera del 75% de acierto real en imágenes nuevas.
* **Resultados Finales:**
  * Parámetros totales: `111430`
  * Accuracy Train: `0.9426`%
  * Accuracy Val: `0.8241`%
  * Accuracy Test: `0.8419`%
