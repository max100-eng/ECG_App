# 🩺 Análisis de ECG con DenseNet201 (8 Clases)

`SDK: gradio` · `app_file: app.py`

## Descripción

Aplicación web (Gradio) para clasificar imágenes de ECG en **8 condiciones cardíacas** usando un modelo basado en **DenseNet201**. Permite subir una imagen de ECG, preprocesarla y obtener la predicción con probabilidades por clase.

## Características

* Interfaz web sencilla con Gradio.
* Preprocesado estándar de imágenes (rescalado, normalización).
* Carga de modelo (PyTorch / TensorFlow — adaptar según implementación).
* Salida: label más probable + probabilidades por clase.
* Lista de las 8 clases (ajustar según tu dataset):

  1. Normal
  2. Fibrilación auricular
  3. Taquicardia supraventricular
  4. Bloqueo AV
  5. Bloqueo de rama
  6. Extrasístole
  7. Infarto agudo
  8. Otra arritmia

> **Nota:** modifica las clases según tu anotación real.

---

## Requisitos

* Python 3.8+
* pip packages:

  * `gradio`
  * `torch` (o `tensorflow` si tu modelo es TF)
  * `torchvision` (si usas PyTorch)
  * `Pillow`
  * `numpy`
  * `opencv-python` (opcional, para preprocesado avanzado)

Ejemplo `requirements.txt` mínimo:

```
gradio
torch
torchvision
Pillow
numpy
opencv-python
```

---

## Estructura recomendada

```
mi-ecg-app/
├── app.py              # Gradio app (interfaz)
├── model.py            # Funciones de carga/modelo (opcional)
├── preprocess.py       # Funciones de preprocesado
├── weights/
│   └── densenet_ecg.pth
├── README.md
└── requirements.txt
```

---

## app.py — ejemplo mínimo (PyTorch)

Copia y adapta este ejemplo a tu modelo y clases:

```python
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import gradio as gr
from torchvision import models

# --- Configuración de clases ---
CLASSES = [
    "Normal",
    "Fibrilación auricular",
    "Taquicardia supraventricular",
    "Bloqueo AV",
    "Bloqueo de rama",
    "Extrasístole",
    "Infarto agudo",
    "Otra arritmia",
]

# --- Cargar modelo (ejemplo DenseNet201) ---
def load_model(weights_path="weights/densenet_ecg.pth", device="cpu"):
    model = models.densenet201(pretrained=False)
    # Ajustar la salida al número de clases
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

DEVICE = "cpu"
MODEL = load_model(weights_path="weights/densenet_ecg.pth", device=DEVICE)

# --- Preprocesado ---
def preprocess_image(img: Image.Image):
    # img: PIL Image (RGB o L)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
        
```




