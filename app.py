import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===============================================
# 1. CONFIGURACIÓN Y CARGA DEL MODELO
# ===============================================

# Definición de las 8 clases de salida
# NOTA: El modelo DENSENET201 DEBE haber sido entrenado con estas 8 clases.
CLASS_NAMES = [
    "Normal",
    "Infarto Agudo",
    "Infarto Antiguo",
    "Fibrilación Auricular",
    "Bloqueo de Rama Izquierda",
    "Bloqueo de Rama Derecha",
    "Extrasístole Auricular",
    "Extrasístole Ventricular"
]

# Configuración del dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# app.py (Fragmento de la función load_model)
# ...

# app.py (Fragmento de la función load_model)

def load_model(model_path: str = "densenet_model.pth"):
    """Carga los pesos del modelo DenseNet201 entrenado."""
    
    # 1. Definir la arquitectura base
    model = models.densenet201(weights=None) 
    
    # 2. RECONSTRUIR LA CAPA CLASIFICADORA (LA CORRECCIÓN CRÍTICA)
    num_ftrs = model.classifier.in_features
    
    # Intentamos reconstruir el clasificador como un nn.Sequential
    # Basado en las claves 'classifier.0' y 'classifier.3'
    
    # NOTA: Debemos adivinar la dimensión intermedia (ej. 512, 1024, 2048).
    # Usaremos 512 como un valor común. Si esto falla, prueba 1024 o el valor que usaste en el entrenamiento.
    HIDDEN_DIM = 512 

    model.classifier = nn.Sequential(
        # Capa 0: Linear. De la salida de DenseNet (num_ftrs) a una dimensión oculta.
        nn.Linear(num_ftrs, HIDDEN_DIM), 
        # Capa 1: Activación (implícita si no tiene pesos, pero la incluimos para claridad).
        nn.ReLU(),
        # Capa 2: Dropout (implícita si no tiene pesos, pero la incluimos para claridad).
        nn.Dropout(0.2), 
        # Capa 3: Linear. De la dimensión oculta a las 8 clases finales.
        nn.Linear(HIDDEN_DIM, len(CLASS_NAMES)) 
    )

    # 3. Cargar los pesos entrenados
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Modelo cargado con éxito desde {model_path} en {device}.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo (Asegúrate de que la arquitectura del clasificador sea correcta): {e}")
        return None 

# ... (El resto del código sigue igual)

# ===============================================
# 2. FUNCIÓN DE PREDICCIÓN
# ===============================================

# Definición de las transformaciones necesarias para el pre-procesamiento
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Valores de normalización estándar para modelos pre-entrenados
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    """Realiza la inferencia en la imagen del ECG."""
    if ecg_model is None:
        # Devuelve un mensaje de error legible en la UI de Gradio
        return {"ERROR: Modelo no cargado (Falta .pth)": 1.0} 

    # 1. Preprocesar la imagen
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # 2. Inferir sin calcular gradientes (más rápido)
    with torch.no_grad():
        output = ecg_model(img_tensor)

    # 3. Post-procesar (Convertir a probabilidades)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Crear el diccionario de resultados para Gradio
    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }
    return results

# ===============================================
# 3. INTERFAZ GRADIO
# ===============================================

# Asignamos la interfaz a la variable 'demo'
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Sube una imagen de ECG"),
    # Muestra las 5 clases con mayor probabilidad
    outputs=gr.Label(num_top_classes=5, label="Clasificación del Modelo"), 
    title="Análisis de ECG con IA: 8 Clases de Diagnóstico",
    description="Sube una imagen de tu electrocardiograma. El modelo clasifica en 8 condiciones cardíacas: Normal, Infarto Agudo, Infarto Antiguo, Fibrilación Auricular, Bloqueo de Rama Izquierda, Bloqueo de Rama Derecha, Extrasístole Auricular y Extrasístole Ventricular.",
    allow_flagging="auto"
)

# ===============================================
# 4. LANZAMIENTO
# ===============================================

if __name__ == "__main__":
    # Usa demo.launch() para iniciar la aplicación web
    # share=True proporciona un enlace público temporal
    demo.launch(share=True)