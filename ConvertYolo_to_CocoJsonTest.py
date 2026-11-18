# https://rfdetr.roboflow.com/learn/train/
# IA Google
import os
import json
from PIL import Image
from shapely.geometry import Polygon

def yolo_seg_to_coco(image_dir, label_dir, output_json_path, classes_list):
    """
    Convierte anotaciones YOLO de segmentación (polígonos) a formato COCO JSON.
    
    :param image_dir: Carpeta que contiene las imágenes
    :param label_dir: Carpeta que contiene los archivos .txt de YOLOv8/v11-seg
    :param output_json_path: Ruta del archivo JSON resultante
    :param classes_list: Lista con los nombres de las clases en orden
    """
    
    coco_dataset = {
        "info": {"description": "Dataset de segmentación convertido de YOLO a COCO"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 1. Crear categorías
    for i, class_name in enumerate(classes_list):
        coco_dataset["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })
        
    annotation_id = 1
    image_id = 1
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
    
    print(f"Procesando {len(image_files)} imágenes con polígonos de segmentación...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(label_dir, label_file)
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error al abrir la imagen {image_file}: {e}")
            continue
            
        # Añadir imagen al JSON
        coco_dataset["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })
        
        if not os.path.exists(label_path):
            image_id += 1
            continue
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7: # Un polígono mínimo necesita 3 puntos (class + 6 coordenadas)
                continue
                
            class_idx = int(parts[0])
            coords_normalizadas = list(map(float, parts[1:]))
            
            # 2. Desnormalizar los puntos del polígono y agruparlos
            polygon_pixels = []
            polygon_points_pairs = [] # Para Shapely [(x1,y1), (x2,y2)...]
            
            for k in range(0, len(coords_normalizadas), 2):
                abs_x = coords_normalizadas[k] * width
                abs_y = coords_normalizadas[k+1] * height
                
                polygon_pixels.append(round(abs_x, 2))
                polygon_pixels.append(round(abs_y, 2))
                polygon_points_pairs.append((abs_x, abs_y))
            
            # Garantizar que el polígono sea válido usando Shapely
            if len(polygon_points_pairs) < 3:
                continue
            
            poly_obj = Polygon(polygon_points_pairs)
            if not poly_obj.is_valid:
                poly_obj = poly_obj.buffer(0) # Intenta reparar polígonos auto-intersecados
                
            # 3. Calcular Bounding Box [x_min, y_min, width, height] a partir del polígono
            min_x, min_y, max_x, max_y = poly_obj.bounds
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            
            # 4. Calcular el área exacta del polígono geométrico
            area = poly_obj.area
            
            # Añadir la anotación con la máscara de segmentación
            coco_dataset["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_idx,
                "bbox": [round(min_x, 2), round(min_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(area, 2),
                "segmentation": [polygon_pixels], # Formato COCO espera una lista de listas de polígonos
                "iscrowd": 0
            })
            annotation_id += 1
            
        image_id += 1

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(coco_dataset, json_file, indent=4, ensure_ascii=False)
        
    print(f"¡Conversión de segmentación completada! Guardado en: {output_json_path}")

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    CARPETA_IMAGENES = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/images"
    CARPETA_ETIQUETAS_YOLO = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/labels"
    ARCHIVO_COCO_SALIDA = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/images/_annotations.coco.json"
    
    # Lista de tus clases en orden
    MIS_CLASES = ["healthy_eye", "leukocoria"] 
    
    yolo_seg_to_coco(
        image_dir=CARPETA_IMAGENES,
        label_dir=CARPETA_ETIQUETAS_YOLO,
        output_json_path=ARCHIVO_COCO_SALIDA,
        classes_list=MIS_CLASES
    )

    from pathlib import Path

    # Definir la ruta del directorio que deseas crear
    nuevo_directorio = Path("Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/images/test")

    try:
        # parents=True crea las carpetas intermedias si no existen
        # exist_ok=True evita errores si la carpeta ya estaba creada
        nuevo_directorio.mkdir(parents=True, exist_ok=True)
        print(f"Directorio listo en: '{nuevo_directorio.resolve()}'")
    except PermissionError:
        print(f"Error: No tienes permisos para crear el directorio en '{nuevo_directorio}'.")
    except OSError as e:
        print(f"Error del sistema operativo al crear el directorio: {e}")

    import shutil
    

    # Definir las rutas de origen y destino
    origen = Path("Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/images")
    destino = Path("Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/images/test")

    # Extensiones permitidas (en minúsculas y con punto)
    extensiones_validas = {".jpg", ".json"}

    try:
        # Verificar que el origen exista y sea un directorio
        if not origen.is_dir():
            raise FileNotFoundError(f"El directorio de origen '{origen}' no existe.")

        # Crear el directorio de destino si aún no existe
        destino.mkdir(parents=True, exist_ok=True)

        archivos_copiados = 0

        # Recorrer los elementos en la raíz del directorio de origen
        for elemento in origen.iterdir():
            # Verificar que sea un archivo y que su extensión esté en la lista permitida
            if elemento.is_file() and elemento.suffix.lower() in extensiones_validas:
                shutil.copy2(elemento, destino / elemento.name)
                archivos_copiados += 1

        print(f"Se han copiado {archivos_copiados} archivos a '{destino}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except PermissionError:
        print("Error: No tienes permisos suficientes para realizar esta operación.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
