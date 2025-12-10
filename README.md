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

## 📂 Estructura del Proyecto

```text
├── dataset/                # Imágenes de prueba para validar el sistema
│   └── test/images/        # Conjunto de imágenes de test
├── modelos/                # Pesos de los modelos entrenados
│   ├── yolo11n.pt          # Modelo base (Personas)
│   ├── yolo11n_train_v1.pt # Modelo custom entrenado (Chalecos)
│   └── yolo11s_train_v1.pt # Modelo custom entrenado (Chalecos)
├── entrenamientos/         # (Opcional) Logs y gráficas del entrenamiento
├── main.py                 # Script principal de ejecución
├── entrenamiento_colab.ipynb # Notebook usado para entrenar el modelo en la nube
├── comparativa_modelos.ipynb # Notebook para analizar métricas entre modelos
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación
