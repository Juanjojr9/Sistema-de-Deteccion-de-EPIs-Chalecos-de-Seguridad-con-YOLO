# 🛡️ Sistema de Detección de EPIs (Chalecos de Seguridad) con YOLOv11

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-green?style=for-the-badge&logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Completado-success?style=for-the-badge)

Este repositorio contiene el proyecto final del curso de **Visión por Computador con IA**. 

El objetivo es desarrollar un sistema capaz de procesar imágenes para detectar trabajadores y verificar si cumplen con la normativa de seguridad (llevar puesto el chaleco reflectante) utilizando Inteligencia Artificial.

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
*   **Modelo B (Chalecos):** Se utiliza un modelo personalizado entrenado específicamente para detectar la clase `safety_vest`.

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

## 📊 Resultados del Entrenamiento y Comparativa

Se han entrenado y comparado tres versiones del modelo para encontrar el equilibrio óptimo entre velocidad y precisión. El entrenamiento se realizó en **Google Colab (T4 GPU)**.

### Tabla de Métricas (Validación)

| Modelo | Arquitectura | Optimizador | mAP@50 | mAP@50-95 | Inferencia (T4 GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Modelo v1.0** | YOLOv11 Nano | Auto (SGD) | 91.0% | 57.9% | **2.2 ms**  |
| **Modelo v2.0** | YOLOv11 Small | Auto (SGD) | 90.7% | **58.5%**  | 4.7 ms |
| **Modelo v3.0** | YOLOv11 Small | AdamW | **91.4%** | 57.7% | 4.8 ms |

### 🏆 Modelo Seleccionado: Modelo v2.0 (Small SGD)

Se ha seleccionado el **Modelo v2.0** para el despliegue final por las siguientes razones:
1.  **Mayor Precisión Estricta:** Ofrece el mejor rendimiento en `mAP@50-95` (0.683 específicamente para la clase `safety_vest`), lo que garantiza que las cajas delimitadoras se ajustan mejor al objeto.
2.  **Robustez:** Al utilizar el optimizador SGD (por defecto en `auto`), demostró una mejor generalización comparado con AdamW (v3.0) en este dataset específico.
3.  **Velocidad Aceptable:** Aunque es más lento que el Nano, 4.7ms por imagen permite procesamiento en tiempo real (>100 FPS), suficiente para vigilancia en obra.

---

## 📂 Archivos Clave y Estructura

El proyecto se organiza de la siguiente manera:

*   **`main.py`**:  
    Script principal de Python. Contiene la lógica de detección, el algoritmo de intersección y la visualización de resultados (ventanas con recuadros verdes/rojos).
*   **`entrenamiento_colab.ipynb`**:  
    Notebook de Jupyter utilizado en Google Colab para entrenar los modelos. Incluye la configuración del entorno, descarga del dataset y ejecución del entrenamiento con GPU.
*   **`comparacion.ipynb`**:  
    Notebook utilizado para cargar los 3 modelos entrenados, validarlos contra el conjunto de test y generar las gráficas y tablas comparativas de rendimiento.
*   **`modelos/`**:  
    Carpeta que contiene los pesos entrenados (`.pt`).
    *   `yolo11n.pt`: Modelo base.
    *   `yolo11s_v1.pt`.
    *    `yolo11s_v2.pt`.

```text
Deteccion-EPIs-YOLO/
│
├── dataset/test/images/    # Imágenes de prueba para validar el sistema
├── modelos/                # Pesos de los modelos entrenados
│   ├── yolo11n.pt          
│   └── yolo11s_v2.pt       
├── main.py                 # Script de inferencia
├── entrenamiento_colab.ipynb 
├── comparacion.ipynb       
├── requirements.txt        
└── README.md
```
## ⚠️ Configuración Importante

Antes de ejecutar main.py o comparacion.ipynb, es necesario configurar las rutas para que coincidan con la estructura de carpetas de tu equipo.
*   Abre el archivo main.py.
*   Busca la sección de CONFIGURACIÓN al principio del archivo.
*   Verifica que las variables apuntan a los archivos correctos:
