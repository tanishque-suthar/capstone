from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Text inputs
text_inputs = processor(text="a white car", padding="max_length", return_tensors="pt")
# Create dummy image
img = Image.new('RGB', (224, 224), color = 'white')
img_inputs = processor(images=[img], return_tensors="pt")

# Join inputs
inputs = {}
inputs.update(text_inputs)
inputs.update(img_inputs)

outputs = model(**inputs)

print("Type of outputs:", type(outputs))
print("Has text_embeds:", hasattr(outputs, 'text_embeds'))
if hasattr(outputs, 'text_embeds'):
    print("text_embeds type:", type(outputs.text_embeds))

try:
    tf = model.get_text_features(**text_inputs)
    print("get_text_features type:", type(tf))
except Exception as e:
    print("get_text_features error:", type(e))
