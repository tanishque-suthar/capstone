from transformers import AutoProcessor, AutoModel
import torch

model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = processor(text="a white car", padding="max_length", return_tensors="pt")
print("calling get_text_features...")
text_features = model.get_text_features(**inputs)
print(type(text_features))
print(dir(text_features))
if hasattr(text_features, 'pooler_output'):
    print(type(text_features.pooler_output))
