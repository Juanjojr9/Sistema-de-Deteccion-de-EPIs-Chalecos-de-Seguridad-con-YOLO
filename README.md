# 🛡️ Sistema de Detección de EPIs (Chalecos de Seguridad) con YOLO

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Status](https://img.shields.io/badge/Status-Completado-success)

Este repositorio contiene el proyecto final del curso de **Visión por Computador con IA**. 

El objetivo es desarrollar un sistema capaz de procesar imágenes en tiempo real para detectar trabajadores y verificar si cumplen con la normativa de seguridad (llevar puesto el chaleco reflectante) utilizando Inteligencia Artificial.

---

## 🎯 Objetivos del Proyecto

El sistema integra dos modelos de Deep Learning para realizar las siguientes tareas:

1.  **Detección de Personas:** Localizar a todos los individuos en la imagen.
2.  **Detección de Chalecos:** Identificar los equipos de protección individual (EPIs).
3.  **Lógica de Intersección:** Determinar algorítmicamente si un chaleco detectado pertenece a una persona específica.
4.  **Alerta Visual:** Clasificar y visualizar a los trabajadores en dos estados:
    *   ✅ **CUMPLE:** Lleva chaleco (Cuadro Verde).
    *   ❌ **NO CUMPLE:** No lleva chaleco (Cuadro Rojo + Alerta).

---

## 🛠️ Tecnologías Utilizadas

*   **Ultralytics YOLOv11:** Arquitectura base para la detección de objetos.
*   **Python 3:** Lenguaje de programación principal.
*   **OpenCV:** Para el preprocesamiento de imágenes y visualización de resultados.
*   **Roboflow:** Gestión del dataset y preprocesamiento.
*   **Google Colab (T4 GPU):** Entorno utilizado para el entrenamiento del modelo.

---

## 📚 Dataset Utilizado

Para el entrenamiento del modelo de detección de chalecos, se ha utilizado un dataset público de alta calidad proporcionado por **Roboflow Universe**:

*   **Nombre:** Safety Vests
*   **Autor:** Roboflow Universe Projects
*   **Versión utilizada:** v13
*   **Enlace:** [Ver Dataset en Roboflow](https://universe.roboflow.com/roboflow-universe-projects/safety-vests/dataset/13)

Este dataset fue exportado en formato **YOLOv11** y contiene imágenes variadas de entornos de construcción y fábricas, lo que garantiza una buena generalización del modelo.

---

## 🧠 Arquitectura y Metodología

El núcleo del sistema (`main.py`) opera mediante una **arquitectura de doble modelo** secuencial:

### 1. Modelos de Inferencia
*   **Modelo A (Personas):** Se utiliza `yolo11n.pt` preentrenado en COCO para detectar la clase `person`. Esto garantiza generalización en la detección de humanos.
*   **Modelo B (Chalecos):** Se utiliza un modelo personalizado (`yolo11n_train_v1.pt`) entrenado específicamente para detectar la clase `safety_vest`.

### 2. Lógica de Negocio (Intersection over Union)
Para evitar falsos positivos (ej. detectar un chaleco colgado en una silla), el sistema aplica lógica geométrica:
1.  Se extraen las cajas delimitadoras (*bounding boxes*) de personas y chalecos.
2.  Se calcula la **Intersección sobre el Área del Chaleco**:
    $$ \text{Overlap} = \frac{\text{Área Intersección}}{\text{Área del Chaleco}} $$
3.  Si la superposición supera el **Umbral (IoU > 0.5)**, se considera que la persona *lleva puesto* el chaleco.

### 3. Filtrado de Falsos Positivos
Se implementan filtros estrictos para limpiar la detección:
*   Filtro por **Clase**: Solo se aceptan detecciones de la clase `1` (Safety Vest), ignorando la clase `0` (No Vest) del dataset para evitar conflictos.
*   Filtro por **Confianza**: Se requiere una certeza > 60% para considerar un chaleco válido.

---

---

## 📂 Estructura del Proyecto

```text
├── dataset/                # Imágenes para entrenar, validad y testear
│   
├── modelos/                # Pesos de los modelos entrenados
│   ├── yolo11n.pt          # Modelo base (Personas)
│   ├── yolo11n_train_v1.pt # Modelo custom entrenado (Chalecos)
│   └── yolo11s_train_v1.pt # Modelo custom entrenado (Chalecos)
├── entrenamientos/         # (Opcional) Logs y gráficas del entrenamiento
├── main.py                 # Script principal de ejecución
├── entrenamiento_colab.ipynb # Notebook usado para entrenar el modelo en la nube
├── comparativa.ipynb # Notebook para analizar métricas entre modelos
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación
