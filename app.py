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
model = AutoModelForImageClassification.from_pretrained(checkpoint)

def process_image(image_data):
    # Convert base64 image to PIL Image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))

    # Define preprocessing steps
    size = (224, 224)  # Assuming the size used in the training notebook
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example values, adjust if needed
    preprocess = Compose([
        RandomResizedCrop(size),  # Use the size from the training notebook
        ToTensor(),
        normalize
    ])

    # Apply preprocessing
    image = preprocess(image.convert("RGB")).unsqueeze(0)  # Add batch dimension

    # Process image using your model
    inputs = {"pixel_values": image}
    with torch.no_grad():
        outputs = model(**inputs)

    # Get prediction
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_class = model.config.id2label[str(predicted_class_idx)]
    confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][predicted_class_idx].item()

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2%}"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json['image']
        result = process_image(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 