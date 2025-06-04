# Sistema de Detección de Plagio en Java

Un sistema completo para detectar plagio en código Java utilizando múltiples técnicas:
- TF-IDF (Frecuencia de Término - Frecuencia Inversa de Documento)
- AST-CC (Componentes Característicos de Árboles de Sintaxis Abstracta)
- Aprendizaje automático con Regresión Logística
- Enfoque híbrido que combina varios métodos

## Instalación

```bash
# Clona el repositorio

# Instala las dependencias requeridas
pip install -r requirements.txt

# Estructura del proyecto
```
```plaintext
├── astcc.py                # Detector de plagio basado en AST
├── karia.py                # Detector basado en TF-IDF y ML
├── main.py                 # Utilidades para comparación
├── data/                   # Directorio del conjunto de datos
│   └── splits/             # Subconjuntos divididos (entrenamiento/validación/prueba)
├── csv/                    # Directorio de salida de resultados en CSV
├── images/                 # Directorio de salida de visualizaciones 
└── models/                 # Archivos de modelos guardados
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación del proyecto
```


## Métodos de Detección

### TF-IDF
Convierte el código a representaciones vectoriales usando la frecuencia de términos y compara similitud usando distancia coseno.

### AST-CC
Analiza la estructura del código mediante árboles de sintaxis abstracta, detectando similitudes estructurales incluso si cambian los nombres de variables o el formato.

### Enfoque basado en ML
Utiliza características TF-IDF con regresión logística para aprender patrones de plagio a partir de ejemplos etiquetados.

### Enfoque Híbrido
Combina predicciones de varios métodos, optimizando los pesos entre TF-IDF y AST-CC para mejorar la precisión general.

---

## Resultados

Los resultados se guardan en directorios estructurados:

- `csv/`: Contiene resultados detallados en formato CSV
- `images/`: Contiene visualizaciones como:
    - Matrices de confusión
    - Curvas ROC
    - Gráficas de importancia de características

---

## Métricas de Evaluación

El sistema evalúa el desempeño usando:

- Precisión (Accuracy)
- Precisión / Recall / F1-score
- ROC-AUC
- Matrices de confusión

