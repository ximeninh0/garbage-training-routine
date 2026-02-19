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
    # Iniciar o treinamento
    # Gemini: 
    results = model.train(
        # ==========================================
        # 1. PARÂMETROS BÁSICOS E DE HARDWARE
        # ==========================================
        data=file, # Certifique-se de que os caminhos aqui estão corretos!
        epochs=150,           # 150 a 300 é um bom ponto de partida. O EarlyStopping ajuda a parar antes se necessário.
        patience=50,          # Para o treino se não houver melhora nas métricas por 50 épocas.
        imgsz=640,            # Resolução padrão. Se os objetos forem muito pequenos, pode testar 1280 (mas consome mais VRAM).
        batch=8,             # Com a RTX 5070, 16 ou 32 deve rodar liso. Se der erro de CUDA, reduza para 8 ou use batch=-1 para AutoBatch.
        device=0,             # Força o uso da sua GPU (RTX 5070).
        workers=2,            # Threads de CPU para carregar os dados. Reduza se a CPU gargalar.
        project='garbage-detector',
        name='640-m',

        # ==========================================
        # 2. PARÂMETROS DE AUGMENTATION RECOMENDADOS
        # ==========================================
        # Rotação e Orientação: Lixo cai em qualquer posição.
        degrees=45.0,         # Rotaciona as imagens em até 45 graus.
        fliplr=0.5,           # 50% de chance de espelhar horizontalmente (padrão).
        flipud=0.5,           # 50% de chance de espelhar verticalmente (essencial para lixo visto de cima).
        
        # Deformação e Escala: Latas e garrafas amassam; o lixo pode estar perto ou longe.
        scale=0.5,            # Varia a escala das imagens em +/- 50%.
        shear=5.0,            # Distorce a imagem levemente (ajuda a simular objetos amassados).
        perspective=0.0001,   # Simula fotos tiradas de ângulos diferentes.
        
        # Cor e Iluminação: Plástico e metal dão muito reflexo; a luz do ambiente varia muito.
        hsv_h=0.015,          # Pequena variação de matiz.
        hsv_s=0.7,            # Variação moderada de saturação (ajuda com reflexos de luz em plástico/metal).
        hsv_v=0.4,            # Variação de brilho/valor (crucial para simular lixeiras escuras ou luz do sol).
        
        # Técnicas de Composição: Ótimo para simular lixeiras cheias (objetos sobrepostos e cortados).
        mosaic=1.0,           # Combina 4 imagens em 1 (excelente para detectar objetos em ambientes caóticos).
        mixup=0.15,           # Sobrepõe imagens levemente (ajuda o modelo a aprender contornos fortes).
        copy_paste=0.1        # Copia objetos e cola em outros lugares da imagem.
    )

# GPT:
#  results = model.train(
#     data="data.yaml",
#     epochs=150,
#     imgsz=640,
#     batch=32,
#     device=0,

#     optimizer="AdamW",
#     lr0=0.0015,
#     lrf=0.01,
#     weight_decay=0.0005,
#     warmup_epochs=3,

#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,

#     degrees=15,
#     translate=0.1,
#     scale=0.5,
#     shear=2,

#     flipud=0.1,
#     fliplr=0.5,

#     mosaic=1.0,
#     mixup=0.2,
#     copy_paste=0.1,

#     workers=8,
#     cache=True
# )

