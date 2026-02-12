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

Los datos presentan variaciones en iluminación y escala, lo que supone un reto realista para el entrenamiento de modelos de visión artificial.

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
