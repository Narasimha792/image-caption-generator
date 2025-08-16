from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/caption', methods=['POST'])
def caption_image():
    file = request.files['image']
    raw_image = Image.open(file).convert("RGB")
    prompt = request.form.get('prompt', '')
    if prompt:
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(raw_image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)