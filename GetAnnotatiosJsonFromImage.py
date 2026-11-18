# IA de Google preguntando "python en un coco json, conociendo el nombre de una imagen, como puedo obtener sus labels"

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

# 1. Cargar el archivo COCO JSON

#coco = COCO("Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/images/_annotations.coco.json")
#coco = COCO("Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb\\images\\test\\_annotations.coco.json")

def GetImageLabeled(dirname, target_image_name, filename_image_name):

    coco = COCO(dirname+"\\_annotations.coco.json")
    print(target_image_name)

    # 2. Nombre de la imagen que buscas
    #target_image_name = "53_png.rf.6e9caa09abbdc6ba7028611a0adf2cec.jpg"
    img = Image.open(target_image_name).convert("RGB")

    # -----------------------------
    # 5. Draw Results
    # -----------------------------
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()


    # 3. Buscar el ID de la imagen
    img_ids = coco.getImgIds()
   
    img_infos = coco.loadImgs(img_ids)
    #print(img_infos)
    img_id = next(
    (img["id"] for img in img_infos if img["file_name"] == filename_image_name),
    None,
    )

    #print(img_id)
    

    if img_id is not None:
        # 4. Obtener las anotaciones asociadas a la imagen
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        print(f"Detected in '{target_image_name}':")

        # 5. Iterar sobre cada anotación para extraer la etiqueta y su bbox
        for ann in anns:
            # Obtener el nombre de la categoría usando su id
            cat_info = coco.loadCats(ann["category_id"])[0]
            label_name = cat_info["name"]

            # COCO format: [x_min, y_min, width, height]
            bbox = ann["bbox"]

            
            x_min, y_min, width, height = bbox

            print(f"- Labeled: {label_name} | BBox [x_min, y_min, ancho, alto]: {bbox}")

            """
            x1=int( x_min - width/2)
            x2=int( x_min + width/2)
            y1=int(y_min - height/2)
            y2=int(y_min - height/2)
            """

            x1=int( x_min )
            x2=int( x_min + width)
            y1=int(y_min)
            y2=int(y_min + height)
            
            
          
            text = f"{label_name}"
            
            if label_name=="healthy_eye":
                color="red"
            else:
                color="blue"
           
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
            text_size=draw.textbbox((0,0),text=text, font=font)
                
            draw.rectangle(((x1, y1 - text_size[1]), (x1 + text_size[0], y1)), fill=color)
                
                
            draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)
            
    else:
            print(f"No image  '{target_image_name}' in dataset.")
            
    return img

#target_image_name = "53_png.rf.6e9caa09abbdc6ba7028611a0adf2cec.jpg"
#target_image_name="Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/test/images/21_png.rf.81f23a20c5b709e0cdc97290ed38c01b.jpg"
#filename_image_name="21_png.rf.81f23a20c5b709e0cdc97290ed38c01b.jpg"
#img=GetImageLabeled(target_image_name, filename_image_name)
#import matplotlib.pyplot as plt    
#plt.imshow(img)
#plt.axis('off')  # Hide axes for a cleaner look
#plt.title(filename)
#plt.show()
    
