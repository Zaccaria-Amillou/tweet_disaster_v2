import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


nltk.download('stopwords')
nltk.download('wordnet')



# Preprocessing
def preprocess(text):
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

# Load the tokenizer
with open('tokenizer/tokenizer_l_glo.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('model/model.h5')

st.title('Disaster Tweet Classifier')

# Add a description
st.write("""
You can enter a tweet in english and the model will analyze it and tell if it refers to a disaster or not.
""")

# Input text box for user to enter a tweet
tweet_input = st.text_input("Enter a tweet:")

if st.button("Predict"):
    if tweet_input:
        # Preprocess the input
        data_prepocess = [preprocess(tweet_input)]
        
        # Tokenize and pad the input
        data_tok = pad_sequences(tokenizer.texts_to_sequences(data_prepocess), maxlen=100)
        
        # Make a prediction
        prediction = model.predict(data_tok)
        
        # Convert the prediction to 'POSITIVE' or 'NEGATIVE'
        st.session_state.sentiment = 'This tweet talks about a disaster ! ' if prediction[0][0] < 0.5 else 'Not a disaster tweet, you are safe'
        
        # Display the prediction
        st.write("Prediction:", st.session_state.sentiment)
    else:
        st.write("Please enter a tweet before predicting.")


