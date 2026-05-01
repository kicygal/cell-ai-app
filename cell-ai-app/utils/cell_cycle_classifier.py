import torch
from torchvision import models, transforms
from PIL import Image
import os
import requests

MODEL_PATH = "cell_cycle_resnet18_best.pth"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1gdhdgbkXzRpr3YR-9noJsF0Dnc6kvnnN"
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = ["anaphase", "interphase", "metaphase", "prophase", "telophase"]
image_size = 224

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_cell_stage(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    predicted_class = class_names[pred_idx.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent