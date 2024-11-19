from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
from torchvision.transforms import Compose, RandomResizedCrop, Normalize, ToTensor
from transformers import AutoModelForImageClassification

app = Flask(__name__)

# Load your model
checkpoint = "SABR22/ViT-threat-classification-v2"
model = AutoModelForImageClassification.from_pretrained(checkpoint, device_map='auto')

def process_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise RuntimeError(f"Error decoding or opening image: {e}")

    try:
        size = (224, 224)
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        preprocess = Compose([
            RandomResizedCrop(size),
            ToTensor(),
            normalize
        ])
        image = preprocess(image.convert("RGB")).unsqueeze(0).to(model.device)  # Move tensor to GPU
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing: {e}")

    try:
        inputs = {"pixel_values": image}
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][predicted_class_idx].item()

        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json['image']
        print("Got image data")
        result = process_image(image_data)
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)