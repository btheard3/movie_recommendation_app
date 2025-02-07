import streamlit as st
import pandas as pd
import numpy as np
import gdown
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz

# Google Drive File IDs (Updated)
FILE_IDS = {
    "ratings": "1yL2oyJMnUfTA_Q0G-Q0bJd_DwwGXzS3_",  
    "movies": "1WwwYXAMa83hyWNNQ6j9z_glg54XhMy48",  
    "tags": "139npUSeGyGY-OEg6NkgYXr-gZUxR03Vr",  
    "tfidf_matrix": "1eb6x6aNCXU8U0epZ7L2-axjeqoA1CkTj",  
    "tfidf_vectorizer": "1eb6x6aNCXU8U0epZ7L2-axjeqoA1CkTj",  
    "movies_with_tags": "1n7_KQ3hG9SUICYpf_LpvDbvvYNgi-yai"  
}

# Function to download files from Google Drive
def download_file(file_key, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={FILE_IDS[file_key]}"
        gdown.download(url, output_path, quiet=False)

# Load datasets with caching
@st.cache_data
def load_data():
    download_file("ratings", "data/ratings.csv")
    download_file("movies", "data/movies.csv")
    download_file("tags", "data/tags.csv")

    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    tags = pd.read_csv("data/tags.csv")

    return ratings, movies, tags

ratings_df, movies_df, tags_df = load_data()

# Load or compute TF-IDF matrix
@st.cache_resource
def compute_tfidf_matrix():
    download_file("tfidf_matrix", "data/tfidf_matrix.npz")
    download_file("tfidf_vectorizer", "data/tfidf_vectorizer.pkl")
    download_file("movies_with_tags", "data/movies_with_tags.csv")

    tfidf_matrix = load_npz("data/tfidf_matrix.npz")
    movies_with_tags = pd.read_csv("data/movies_with_tags.csv")

    with open("data/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    return tfidf_matrix, movies_with_tags, tfidf

tfidf_matrix, movies_with_tags, tfidf = compute_tfidf_matrix()

# Streamlit App
st.title("🎬 Movie Recommendation System")

st.subheader("🔎 Search for Movies by Genre or Tags")
search_query = st.text_input("Enter a movie or genre:")

if search_query:
    search_vector = tfidf.transform([search_query])
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarity_scores)[-10:][::-1]
    recommendations = movies_with_tags.iloc[top_indices]

    st.subheader("🎬 Recommendations")
    for _, row in recommendations.iterrows():
        st.write(f"- **{row['title']}** (Genres: {row['genres']})")
