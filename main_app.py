import streamlit as st
import pandas as pd
import numpy as np
from constants import MODEL_PATH, SCALER_PATH, VECTORIZER_PATH, THRESHOLD
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

def remove_negation_words_from_stopwords(stopwords_list):
    # Load and customize stopwords
    negation_words_list = {'not', 'no', 'nor', "didn't", "wasn't", "isn't", "aren't", "doesn't"}
    modified_stopwords_list = stopwords_list - negation_words_list
    return modified_stopwords_list

# Function to load models and other resources
def load_resources():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, scaler, vectorizer

def preprocess_text(text_input):
    lemmatizer = WordNetLemmatizer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review_list = review.lower().split()
    stop_words = set(stopwords.words("english"))
    modified_stopwords_list = remove_negation_words_from_stopwords(stop_words)
    review_list = [lemmatizer.lemmatize(word) for word in review_list if word not in modified_stopwords_list]
    print("review list ", review_list)
    return " ".join(review_list)


# Predicting sentiment based on input
def predict_sentiment(text_input):
    text_input_preprocessed = preprocess_text(text_input)
    x_prediction = vectorizer.transform([text_input_preprocessed]).toarray()
    x_prediction_scaled = scaler.transform(x_prediction)
    y_prediction_probability = model.predict_proba(x_prediction_scaled)
    print("Processed text:", text_input_preprocessed)
    print("Scaled feature vector:", x_prediction_scaled)
    print("Prediction probabilities:", y_prediction_probability)
    positive_probability = y_prediction_probability[0][1]
    if positive_probability > THRESHOLD:
        return "Positive"
    else:
        return "Negative"

# streamlit UI
st.title('Alexa Reviews Sentiment Prediction project')
st.write('This application predicts the sentiment of the input text using XGBoost model.')

# taking user input
user_input = st.text_area('Enter text here: ', value = "", height = None, max_chars = None, key = None)

if st.button('Predict Sentiment'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f'Predicted Sentiment : {prediction}')
    else:
        st.write('Please enter some text to analyze.')