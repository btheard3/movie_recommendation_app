import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.sparse import save_npz, load_npz
import os

# Cached function to load datasets
@st.cache
def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    tags = pd.read_csv('data/tags.csv')
    return ratings, movies, tags

ratings_df, movies_df, tags_df = load_data()

# Load or compute TF-IDF matrix
if os.path.exists('data/tfidf_matrix.npz'):
    tfidf_matrix = load_npz('data/tfidf_matrix.npz')
else:
    @st.cache
    def compute_tfidf_matrix():
        movie_tags = tags_df.groupby('movieId')['tag'].apply(' '.join).reset_index()
        movies_with_tags = pd.merge(movies_df, movie_tags, on='movieId', how='left')
        movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(movies_with_tags['tag'])
        save_npz('data/tfidf_matrix.npz', matrix)  # Save for future use
        return matrix

    tfidf_matrix = compute_tfidf_matrix()

# Load or handle model
model_path = 'models/recommendation_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
else:
    st.warning("Model not found. Please train the model and save it to models/recommendation_model.h5.")

# Streamlit App
st.title("Enhanced Movie Recommendation System")

# User Feedback
feedback = st.text_area("Tell us about your preferences (e.g., 'I enjoy action-packed thrillers').")
if feedback:
    sentiment = TextBlob(feedback).sentiment
    st.write(f"Sentiment Analysis: Polarity={sentiment.polarity}, Subjectivity={sentiment.subjectivity}")

# Search for Movies
search_query = st.text_input("Search for a movie or genre:")
if search_query:
    search_vector = TfidfVectorizer(stop_words='english').fit_transform([search_query])
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()

    top_indices = np.argpartition(similarity_scores, -10)[-10:]
    recommendations = movies_df.iloc[top_indices]

    st.write("Top matching movies:")
    for _, row in recommendations.iterrows():
        st.write(f"- {row['title']} ({row['genres']})")
