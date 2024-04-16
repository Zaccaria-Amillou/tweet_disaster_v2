import re
import nltk
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
def process_tweets(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+|[-+]?\d*\.\d+|\d+"
    
    # Remove link,user and special characters
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)

# Load your trained model and tokenizer
model = load_model('model/best_model_model.h5')
with open('tokenizer/tokenizer_l_glo.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

st.title('Sentiment Analysis')

def predict_sentiment(model, tokenizer, text):
    # Preprocess the text
    text = process_tweets(text)
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    # Make the prediction
    prediction = model.predict(padded_sequence)
    # Convert the prediction to a sentiment
    sentiment = 'This is a disaster tweet' if prediction[0][0] > 0.5 else 'Not a disaster tweet'
    return sentiment, prediction[0][0]

user_input = st.text_input("Enter a sentence to analyze its sentiment")

if st.button('Predict'):
    sentiment, probability = predict_sentiment(model, tokenizer, user_input)
    st.write(f'Sentiment: {sentiment}, Probability: {probability}')