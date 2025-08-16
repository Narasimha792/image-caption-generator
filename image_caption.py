from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and process image
image_path = "images.jpeg"  # Your image file
raw_image = Image.open(image_path).convert("RGB")

# ----- Choose one: Prompt-based or Free captioning -----

# 1. Prompt-based captioning
prompt = "a photography of"
inputs = processor(raw_image, prompt, return_tensors="pt").to(device)

# 2. Free captioning (uncomment to use)
# inputs = processor(raw_image, return_tensors="pt").to(device)

# Generate caption
output = model.generate(**inputs)

# Decode and print
caption = processor.decode(output[0], skip_special_tokens=True)
print(f"Description: {caption}")
