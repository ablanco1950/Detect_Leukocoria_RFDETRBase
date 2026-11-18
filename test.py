
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np
from PIL import Image

THRESHOLD=0.4

# directory of test images
dirname ="Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb\\images\\test"

classes=['healthy','leukocoria']


from rfdetr import RFDETRNano # Para entrenar solo con cpu

model = RFDETRNano(pretrain_weights="output\checkpoint.pth")
#model = RFDETRNano(pretrain_weights="output\checkpoint_best_total.pth")


from GetAnnotatiosJsonFromImage import GetImageLabeled

import os
import re
def loadimages(dirname):
 
     imgpath = dirname + "\\"      
     
     images = []
     TabFileName=[]   
    
     print("Reading images from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 image = Image.open(filepath)                
                 
                                         
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
                 
     print("Readed " + str(len(images)))
     
     return images, TabFileName

#
# MAIN
#


images, TabFileName=loadimages(dirname)

for i in range(len(images)):

    image=images[i]

    detections = model.predict(image, threshold=THRESHOLD)

    
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            smart_position=True)
    

    detections_image = image.copy()
    
    # 1. Configurar colores: Verde (0: healthy) y Rojo (1: leukocoria)
    custom_palette = sv.ColorPalette(colors=[sv.Color.RED, sv.Color.BLUE])
    bbox_annotator = sv.BoxAnnotator(color=custom_palette)
    label_annotator = sv.LabelAnnotator(
    color=custom_palette, text_color=sv.Color.WHITE
     )

    # 2. Filtrar ambas clases y crear etiquetas con score
    selected_detections = detections[np.isin(detections.class_id, [0, 1])]
    class_names = {0: "healthy", 1: "leukocoria"}

    labels = [
    f"{class_names[class_id]} {confidence:0.2f}"
    for class_id, confidence in zip(
        selected_detections.class_id, selected_detections.confidence
     )
     ]

    # 3. Anotar sobre copia de la imagen
    annotated_image = bbox_annotator.annotate(
    scene=detections_image.copy(), detections=selected_detections
     )
    annotated_image = label_annotator.annotate(
     scene=annotated_image, detections=selected_detections, labels=labels
     )

    # 4. Guardar automáticamente en disco (opcional)
    #cv2.imwrite("deteccion_leukocoria.jpg", annotated_image)

    filename=TabFileName[i]

    labeled_image=GetImageLabeled(dirname, dirname + "/" + filename, filename)
    
    title_plot_labeled= "labeled : " + filename

    # 1. Evita que plt.show() limpie la figura si estás en un script o notebook
    #plt.ion()
    
    #%matplotlib widget para que aparezca las opciones de matplotlib como guardar
    # 5. Mostrar en pantalla
    sv.plot_images_grid(
     #images=[detections_image, annotated_image],
     images=[labeled_image, annotated_image],   
     grid_size=(1, 2),
     #titles=["Original", "Detections (Healthy & Leukocoria)"],
     titles=[title_plot_labeled, "Detections"],
     )

    # 2. Guarda la figura actual en disco antes de cerrarla
    #plt.savefig("resultado_compuesto.png", bbox_inches="tight", dpi=300)
    #plt.close()
    
