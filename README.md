# 🎬 Movie Review Sentiment Analysis (Streamlit App)

## Overview of project
This project is a Movie Review Sentiment Analysis System built using Deep Learning (SimpleRNN) and Streamlit. It classifies movie reviews as Positive or Negative using a pre-trained model on the IMDB dataset.

## The application supports:
-✍️ Manual review input
-📂 Bulk CSV file upload with automatic analysis and visualization

## 🚀 Features
### ✅ Manual Review Classification
- User can type a movie review
- Model predicts:
- Sentiment (Positive / Negative)
- Prediction confidence score

## ✅ CSV File Upload (Bulk Analysis)
- Upload a CSV file with any column name
- Automatically detects the text column
- Analyzes 100+ reviews at once
### ~Displays:
- Total Positive & Negative reviews
- Percentage distribution
- Bar chart visualization

## 🧠 Model Details
- Model Type: SimpleRNN
- Dataset: IMDB Movie Reviews
- Vocabulary Size: 10,000
- Sequence Length: 500
- Framework: TensorFlow / Keras

## 🛠️ Tech Stack
Python 🐍
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Streamlit
