# Disaster Tweet Classifier

This is a Streamlit application that uses a trained machine learning model to classify tweets as either referring to a disaster or not.

## Features

- User-friendly interface for entering a tweet.
- Uses a trained model to predict whether the entered tweet refers to a disaster or not.
- Displays the prediction result to the user.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Zaccaria-Amillou/tweet_disaster_v2.git
   ```

2. Navigate to the project directory:

   ```bash
   cd tweet_disaster_v2
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit tweet_disaster.py
   ```

2. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to view the application.

3. Enter a tweet in the text box and click the "Predict" button to get a prediction.

## Dependencies

- Streamlit
- Numpy
- TensorFlow
- NLTK
- Pickle
- re

## Model

The model used for prediction is a pre-trained model saved as `model/model.h5`. The model was trained on a dataset of tweets, with labels indicating whether the tweet referred to a disaster or not taken from the kaggle competition called **Natural Language Processing with Disaster Tweets** 
link: https://www.kaggle.com/competitions/nlp-getting-started.

The text of the tweets is preprocessed and tokenized before being fed into the model. The preprocessing steps include removing links, special characters, and stopwords, and lemmatizing the words.
