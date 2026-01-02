# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review

VOCAB_SIZE = 10000  # must match model

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        idx = word_index.get(word)

        # idx + 3 must ALSO be < VOCAB_SIZE
        if idx is not None and (idx + 3) < VOCAB_SIZE:
            encoded_review.append(idx + 3)
        else:
            encoded_review.append(2)  # UNKNOWN token

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500,
        padding='pre',
        truncating='pre'
    )

    return padded_review


#streamlit app


import streamlit as st
st.sidebar.title("✅Select Option")
feature = st.sidebar.radio("Choose Input Method" ,
                           ("Manual Review" , "CSV Upload")) 

if feature == "Manual Review":
    st.title("📺Movie Review Sentiment Analyzer")
    st.write('Enter a movie review to classify it as positive or negative.')

# User input
    user_input = st.text_area('Movie Review')

    if st.button('Classify'):
        preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
        prediction=model.predict(preprocessed_input)
        sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
        st.write(f'💡Sentiment: {sentiment}')
        st.write(f'🎯Prediction Score: {prediction[0][0]}')
    else:
        st.write('Please enter a movie review.')


## for upload csv
elif feature == "CSV Upload":
    st.markdown("---")
    st.subheader("📂 Bulk Review Analysis (CSV Upload)")

    uploaded_file = st.file_uploader(
    "Upload CSV file containing movie reviews (any column name allowed)",
    type=["csv"]
    )

    if uploaded_file is not None:
        try:
        # Read CSV
            df = pd.read_csv(uploaded_file)

        # Automatically select the first text column
            text_column = None
            for col in df.columns:
                if df[col].dtype == object:
                    text_column = col
                    break

            if text_column is None:
                st.error("❌ No text column found in the uploaded CSV.")
            else:
                reviews = df[text_column].dropna()

                positive_count = 0
                negative_count = 0

                for review in reviews:
                    processed_review = preprocess_text(str(review))
                    prediction = model.predict(processed_review, verbose=0)

                    if prediction[0][0] > 0.5:
                        positive_count += 1
                    else:
                        negative_count += 1

                total = positive_count + negative_count
                pos_percent = (positive_count / total) * 100
                neg_percent = (negative_count / total) * 100

            # Display results
                st.success("✅ Bulk Sentiment Analysis Completed")
                st.write(f"📌 Column used for analysis: **{text_column}**")
                st.write(f"👍 Positive Reviews: {positive_count} ({pos_percent:.2f}%)")
                st.write(f"👎 Negative Reviews: {negative_count} ({neg_percent:.2f}%)")

            # Visualization
                fig, ax = plt.subplots()
                ax.bar(["Positive", "Negative"], [positive_count, negative_count])
                ax.set_ylabel("Number of Reviews")
                ax.set_title("Sentiment Distribution")

                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error processing the file: {e}")

    
 

