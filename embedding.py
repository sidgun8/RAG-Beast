from transformers import AutoTokenizer, AutoModel
import torch
import sys
import json

class LocalEmbeddingGemma:
    def __init__(self):
        print("Loading EmbeddingGemma locally...")
        self.tokenizer = AutoTokenizer.from_pretrained("./models/embeddinggemma")
        self.model = AutoModel.from_pretrained("./models/embeddinggemma")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean Pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

if __name__ == "__main__":
    # Node.js will send input as JSON string
    input_text = json.loads(sys.argv[1])
    gemma = LocalEmbeddingGemma()
    embedding = gemma.get_embedding(input_text)
    # Return embedding as JSON
    print("🟢 Using EmbeddingGemma locally (CPU)")

