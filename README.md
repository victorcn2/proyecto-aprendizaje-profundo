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

