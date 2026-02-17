from ultralytics import YOLO
import torch

model = YOLO("yolo11s.pt")
file = "data.yaml"

if __name__ == "__main__":
    results = model.train(        
        data=file,
        
        workers=6,       
        device=[0],     
        batch=12,         
        imgsz=512,       
        
        epochs=150,     
        patience=125,     
        
        #testando sem data augmentation primeiro
        # mosaic=0.7,      # Reduz mosaic
        # mixup=0.0,       # Remove mixup
        # copy_paste=0.0,  # Remove copy-paste
        
        # Configurações de projeto
        project='runs/512-s',
        name='17-02',
        exist_ok=True,
        resume=False,
        
        )

