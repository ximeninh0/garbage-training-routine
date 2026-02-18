from ultralytics import YOLO
import torch

model = YOLO("yolo11m.pt")
file = "data.yaml"

print(f"Torch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    results = model.train(        
        data=file,
        
        workers=4,       
        device=[0],     
        batch=32,         
        imgsz=640,       
        
        epochs=250,     
        patience=150,     
        
        #testando sem data augmentation primeiro
        # mosaic=0.7,      # Reduz mosaic
        # mixup=0.0,       # Remove mixup
        # copy_paste=0.0,  # Remove copy-paste
        
        # Configurações de projeto
        project='runs/640-m',
        name='17-02',
        exist_ok=True,
        resume=False,
        
        )

