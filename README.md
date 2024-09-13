Amazon Alexa Reviews - Sentiment Analysis Project

Project Overview
This project aims to analyze Amazon Alexa customer reviews and build a classification model to predict whether the sentiment of a review is positive or negative. The insights from this analysis help companies understand customer feedback, enhance product development, and improve user experiences.

The project uses machine learning models, specifically Random Forest, Decision Trees, and XGBoost, to classify sentiment. The XGBoost model was identified as the best performing model based on its accuracy in both training and testing.

Dataset
The dataset contains:
- 3000 Amazon Alexa product reviews (Echo, Echo Dot, Firestick, etc.)
- Star ratings, review dates, and feedback for each review
- Source: Amazon Alexa Reviews Dataset - Kaggle https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews

Key Steps

1) Data Preprocessing:
  - Removed non-alphabetic characters, converted to lowercase, tokenized, and stemmed words.
  - Vectorized text data using Bag of Words.
  - Exploratory Data Analysis (EDA):

2) Data Visualization
  - Visualized the distribution of ratings and feedback.
  - Generated word clouds to display common positive and negative terms in reviews.

3) Modeling:
  - Implemented three models:
  - Random Forest: 94% accuracy on test data.
  - Decision Trees: 92% accuracy on test data.
  - XGBoost: 95% accuracy on test data (best model).

4) Model Evaluation
  - Evaluated models using AUC-ROC curves and confusion matrix.


5) Model Deployment
  - Deployed the XGBoost model using Streamlit, allowing users to input reviews and get sentiment predictions (Positive/Negative).
  - The model will predict if the sentiment is Positive or Negative based on the provided review.

6) Results
  - XGBoost Model: 95% accuracy on test data
  - Best model for analyzing Amazon Alexa reviews

Project Files
main_app.py: Streamlit application for sentiment prediction
Models/: Directory containing pre-trained models
notebooks/: Jupyter notebooks for data exploration and modeling

Conclusion
This project was successfully built and deployed a sentiment analysis model for Amazon Alexa product reviews. The XGBoost model was selected as the best-performing model, achieving high accuracy on both training and testing data. The model was deployed using Streamlit, enabling real-time sentiment analysis of new reviews.

