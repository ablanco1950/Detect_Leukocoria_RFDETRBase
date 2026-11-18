"""

asking IA Google: "intento entrenar con python un modelo RFDETRBase con custom dataset en un ordenador personal con 16gb de memoria, pero el sistema se cae. Existe alguna version ligera de RFDETRBase
o ajuste de parametros que me pueda servir?"

"""

# dataset https://universe.roboflow.com/leukocoria-detector-nv8ij/leukocoria-detector-rsl8x/dataset/2


import torch
#from rfdetr import RFDETRBase
from rfdetr import RFDETRNano # Para entrenar solo con cpu

# Limita los hilos de CPU para evitar que Windows se congele por completo
torch.set_num_threads(4) 




# https://stackoverflow.com/questions/75517324/runtimeerror-inference-tensors-cannot-be-saved-for-backward-to-work-around-you
torch.inference_mode=torch.no_grad


def main():
        #model = RFDETRBase()
        model = RFDETRNano()
        history = []

        def callback2(data):
                history.append(data)

        model.callbacks["on_fit_epoch_end"].append(callback2)
        
        #model.train(dataset_dir="Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb\\images", epochs=20, batch_size=16, lr=1e-4)
        

        model.train(
            dataset_dir="Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb\\images",
            device="cpu",           # Fuerza el uso explícito de CPU
            epochs=30,              # Empieza con pocas épocas para medir tiempos
            batch_size=1,           # Mínimo consumo de RAM
            grad_accum_steps=16,    # Simula un batch de 16 sin gastar memoria extra
            num_workers=0,          # CRÍTICO en Windows: evita duplicar procesos en RAM
            pin_memory=False,       # No necesario sin GPU
            backbone_lora=True,     # Reduce la carga de entrenamiento
            use_ema=False,          # Ahorra RAM al no duplicar los pesos del modelo
            multi_scale=False,      # Evita picos de memoria por imágenes grandes
            lr=1e-4
        )

        
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(history)

        plt.figure(figsize=(12, 8))

        plt.plot(
                df['epoch'],
                df['train_loss'],
                label='Training Loss',
                marker='o',
                linestyle='-'
        )

        plt.plot(
                df['epoch'],
                df['test_loss'],
                label='Validation Loss',
                marker='o',
                linestyle='--'
        )

        plt.title('Train/Validation Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.show()
       
if __name__ == "__main__":
    main() 
