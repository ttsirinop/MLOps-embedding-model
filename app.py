from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().tolist()

@app.route('/embeddings', methods=['POST'])
def embeddings():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    text = data['text']
    embedding = get_embedding(text)
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
