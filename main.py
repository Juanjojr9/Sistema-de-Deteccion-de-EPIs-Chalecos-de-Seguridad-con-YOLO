import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob 
import random
# ============================================
# CONFIGURACIÓN
# ============================================
BASE_DIR = os.getcwd()
MODELO_PERSONAS_PATH = os.path.join(BASE_DIR, "modelos", "yolo11n.pt")
MODELO_CHALECOS_PATH = os.path.join(BASE_DIR, "modelos", "yolo11s_train_v2.pt") 
DIR_IMAGENES_TEST = os.path.join(BASE_DIR, "dataset", "test", "images")

# Hiperparámetros de Lógica
UMBRAL_CONF_PERSONA = 0.4  # Alta confianza para evitar falsos positivos
UMBRAL_CONF_CHALECO = 0.5  # Un poco más bajo por si son pequeños
UMBRAL_IOU = 0.5          # % del chaleco que debe estar dentro de la persona

# ============================================
# FUNCIÓN DE SUPERPOSICIÓN
# ============================================
def calcular_superposicion_chaleco(box_persona, box_chaleco):
    """
    Calcula qué porcentaje del área del chaleco está contenido dentro
    del bounding box de la persona.
    """
    # Coordenadas de la persona (xp) y chaleco (xc)
    xp1, yp1, xp2, yp2 = box_persona
    xc1, yc1, xc2, yc2 = box_chaleco

    # Calcular coordenadas de la intersección
    x_inter1 = max(xp1, xc1)
    y_inter1 = max(yp1, yc1)
    x_inter2 = min(xp2, xc2)
    y_inter2 = min(yp2, yc2)

    # Calcular área de intersección (si no hay solapamiento, es 0)
    ancho_inter = max(0, x_inter2 - x_inter1)
    alto_inter = max(0, y_inter2 - y_inter1)
    area_interseccion = ancho_inter * alto_inter
    
    # Calcular área total del chaleco
    area_chaleco = (xc2 - xc1) * (yc2 - yc1)
    
    # Evitar división por cero
    if area_chaleco == 0:
        return 0.0
    
    # Retornamos el ratio (0.0 a 1.0)
    return area_interseccion / float(area_chaleco)

# ============================================
# 3. PROCESAMIENTO PRINCIPAL
# ============================================
def main():
    print("--- INICIANDO SISTEMA DE SEGURIDAD ---")

    # A. Cargar Modelos
    try:
        print(f"Cargando modelo personas: {MODELO_PERSONAS_PATH}")
        model_personas = YOLO(MODELO_PERSONAS_PATH)
        
        print(f"Cargando modelo chalecos: {MODELO_CHALECOS_PATH}")
        model_chalecos = YOLO(MODELO_CHALECOS_PATH)
        print("\n🔍 CLASES DEL MODELO DE CHALECOS:")
        print(model_chalecos.names)
        print("-------------------------------------\n")
    except Exception as e:
        print(f"❌ Error crítico cargando modelos: {e}")
        return

    # B. Buscar imágenes de prueba
    # Usamos glob para encontrar todos los .jpg en la carpeta
    patron_busqueda = os.path.join(DIR_IMAGENES_TEST, "*.jpg")
    imagenes_encontradas = glob.glob(patron_busqueda)
    
    # MEJORA: Mezclar la lista para que sean aleatorias cada vez
    random.shuffle(imagenes_encontradas) 

    # Coger las 5 primeras (que ahora serán distintas en cada ejecución)
    imagenes_a_procesar = imagenes_encontradas[:5]

    if not imagenes_a_procesar:
        print(f"❌ No se encontraron imágenes en: {DIR_IMAGENES_TEST}")
        return

    print(f"✅ Se procesarán {len(imagenes_a_procesar)} imágenes.\n")

    # C. Bucle de procesamiento por imagen
    for img_path in imagenes_a_procesar:
        print(f"--> Procesando: {os.path.basename(img_path)}")
        imagen = cv2.imread(img_path)
        
        if imagen is None:
            print("   Error de lectura de imagen.")
            continue

        # ---------------------------------------------------------
        # D. INFERENCIA (Detección)
        # ---------------------------------------------------------
        # Detectar Personas (Clase 0 de COCO es 'person')
        # verbose=False limpia la consola
        res_personas = model_personas.predict(imagen, classes=[0], conf=UMBRAL_CONF_PERSONA, verbose=False)
        
        # Detectar Chalecos (Modelo personalizado)
        res_chalecos = model_chalecos.predict(imagen, conf=UMBRAL_CONF_CHALECO, verbose=False, classes=[1])

        # Extraer coordenadas a listas limpias
        boxes_personas = []
        for box in res_personas[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            boxes_personas.append(coords)

        boxes_chalecos = []
        for box in res_chalecos[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            boxes_chalecos.append(coords)

        print(f"   Personas detectadas: {len(boxes_personas)}")
        print(f"   Chalecos detectados: {len(boxes_chalecos)}")

        # ---------------------------------------------------------
        # E. LÓGICA DE NEGOCIO (Matching Persona-Chaleco)
        # ---------------------------------------------------------
        personas_sin_chaleco = []
        personas_con_chaleco = []

        for persona in boxes_personas:
            tiene_chaleco = False
            
            # Comparamos esta persona contra TODOS los chalecos encontrados
            for chaleco in boxes_chalecos:
                coincidencia = calcular_superposicion_chaleco(persona, chaleco)
                
                # Si el chaleco está suficientemente "dentro" de la persona
                if coincidencia > UMBRAL_IOU:
                    tiene_chaleco = True
                    break # Ya encontramos su chaleco, dejamos de buscar
            
            if tiene_chaleco:
                personas_con_chaleco.append(persona)
            else:
                personas_sin_chaleco.append(persona)

        # ---------------------------------------------------------
        # F. VISUALIZACIÓN DE RESULTADOS
        # ---------------------------------------------------------
        # 1. Dibujar Personas SIN chaleco (ROJO - ALERTA)
        for i, box in enumerate(boxes_chalecos):
            xc1, yc1, xc2, yc2 = box
            confianza = float(res_chalecos[0].boxes.conf[i]) # Obtenemos la confianza real
            
            # Dibujamos caja amarilla
            cv2.rectangle(imagen, (xc1, yc1), (xc2, yc2), (0, 255, 255), 2)
            # Ponemos el % de confianza encima para ver por qué falla
            label_chaleco = f"Vest: {confianza:.2f}"
            cv2.putText(imagen, label_chaleco, (xc1, yc1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for (x1, y1, x2, y2) in personas_sin_chaleco:
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Fondo para el texto para que se lea mejor
            cv2.rectangle(imagen, (x1, y1-30), (x1+160, y1), (0, 0, 255), -1)
            cv2.putText(imagen, "SIN CHALECO", (x1+5, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2. Dibujar Personas CON chaleco (VERDE - OK)
        for (x1, y1, x2, y2) in personas_con_chaleco:
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagen, "OK", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3. (Opcional) Dibujar los chalecos en amarillo fino para depurar
        for (x1, y1, x2, y2) in boxes_chalecos:
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Mostrar estadísticas en consola
        print(f"   [RESULTADO] Cumplen: {len(personas_con_chaleco)} | Incumplen: {len(personas_sin_chaleco)}")

        # Mostrar imagen
        # Redimensionar si es muy grande para la pantalla
        alto, ancho = imagen.shape[:2]
        if alto > 800:
            scale_percent = 800 / alto
            width = int(ancho * scale_percent)
            height = int(alto * scale_percent)
            imagen = cv2.resize(imagen, (width, height))

        cv2.imshow("Sistema de Deteccion EPIs", imagen)
        print("   Presiona cualquier tecla para pasar a la siguiente imagen (o 'q' para salir)...")
        
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n--- PROCESO FINALIZADO ---")

if __name__ == "__main__":
    main()