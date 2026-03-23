from transformers import AutoProcessor, AutoModel
import torch
import sys

def test():
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text_inputs = processor(text="a white car", padding="max_length", return_tensors="pt")
    out = model(**text_inputs)
    print("Text embeds type:", type(out.text_embeds))

if __name__ == "__main__":
    test()
