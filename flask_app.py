from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import numpy as np

# Load tokenizer, label encoder, and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_enc = joblib.load('label_encoder.pkl')

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = SentimentClassifier(vocab_size=10000, embed_dim=128, hidden_dim=128, output_dim=len(label_enc.classes_))
model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    input_tensor = torch.tensor(padded, dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        sentiment = label_enc.inverse_transform([pred])[0]

    return jsonify({"text": text, "predicted_sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
