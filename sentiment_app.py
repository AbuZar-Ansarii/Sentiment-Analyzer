import streamlit as st
import torch
import torch.nn as nn
import pickle
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_enc = joblib.load('label_encoder.pkl')


# Define the model class again
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


# Load model
model = SentimentClassifier(vocab_size=10000, embed_dim=128, hidden_dim=128, output_dim=len(label_enc.classes_))
model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("🧠 Sentiment Analyzer")
st.markdown("Enter a sentence below to detect its sentiment.")

text_input = st.text_area("Your sentence:")

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
        input_tensor = torch.tensor(padded, dtype=torch.long)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            sentiment = label_enc.inverse_transform([pred])[0]

        st.success(f"**Predicted Sentiment:** {sentiment}")
