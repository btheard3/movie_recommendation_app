import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.sparse import save_npz, load_npz
import os

# Load datasets with caching
@st.cache_data
def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    tags = pd.read_csv('data/tags.csv')
    return ratings, movies, tags

ratings_df, movies_df, tags_df = load_data()

# 🔥 New: Keyword Mapping for Better Genre Recognition
GENRE_KEYWORDS = {
    "scary": "Horror",
    "funny": "Comedy",
    "romantic": "Romance",
    "space": "Sci-Fi",
    "future": "Sci-Fi",
    "adventure": "Action|Adventure",
    "hero": "Action|Superhero",
    "detective": "Mystery|Thriller",
    "historical": "History|Drama",
    "sad": "Drama",
    "epic": "Fantasy|Adventure",
}

# Load or compute TF-IDF matrix and vectorizer with caching
@st.cache_resource
def compute_tfidf_matrix():
    tags_df['tag'] = tags_df['tag'].fillna('')
    movie_tags = tags_df.groupby('movieId')['tag'].apply(' '.join).reset_index()
    movies_with_tags = pd.merge(movies_df, movie_tags, on='movieId', how='left')
    movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

    # Initialize and fit TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies_with_tags['tag'])

    # Save both the TF-IDF matrix and the vectorizer for future use
    save_npz('data/tfidf_matrix.npz', matrix)
    with open('data/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    movies_with_tags.to_csv('data/movies_with_tags.csv', index=False)
    
    return matrix, movies_with_tags, tfidf

# Ensure precomputed files exist; otherwise, recompute
if os.path.exists('data/tfidf_matrix.npz') and os.path.exists('data/movies_with_tags.csv') and os.path.exists('data/tfidf_vectorizer.pkl'):
    tfidf_matrix = load_npz('data/tfidf_matrix.npz')
    movies_with_tags = pd.read_csv('data/movies_with_tags.csv')
    
    # Load the saved TF-IDF vectorizer
    with open('data/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
else:
    tfidf_matrix, movies_with_tags, tfidf = compute_tfidf_matrix()

# Load the recommendation model with caching
@st.cache_resource
def load_recommendation_model():
    model_path = 'models/recommendation_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    else:
        st.error("Model file not found. Please train and save the model.")
        return None

model = load_recommendation_model()

# Streamlit App
st.title("🎬 Enhanced Movie Recommendation System")

# 🎭 **User Feedback for Personalized Recommendations**
st.subheader("🎭 Enter Your Movie Preferences")
feedback = st.text_area("Describe the type of movies you like (e.g., 'I enjoy action-packed thrillers'):")

if feedback:
    # **Sentiment Analysis**
    sentiment = TextBlob(feedback).sentiment
    st.write(f"**Sentiment Analysis:** Polarity = {sentiment.polarity:.2f}, Subjectivity = {sentiment.subjectivity:.2f}")

    # **Map Feedback Keywords to Genres**
    detected_genres = []
    for word, mapped_genre in GENRE_KEYWORDS.items():
        if word in feedback.lower():
            detected_genres.append(mapped_genre)

    if detected_genres:
        filtered_movies = movies_df[movies_df['genres'].str.contains('|'.join(detected_genres), na=False)]
        recommended_movies = filtered_movies.sample(n=min(10, len(filtered_movies))) if not filtered_movies.empty else pd.DataFrame()

        st.subheader("🎥 Personalized Movie Recommendations")
        if not recommended_movies.empty:
            for _, row in recommended_movies.iterrows():
                st.write(f"- **{row['title']}** (Genres: {row['genres']})")
        else:
            st.warning("No movies found matching your feedback. Try different wording.")
    else:
        st.warning("No genres detected in your feedback. Try mentioning specific movie genres.")

# 🔎 **Movie Search**
st.subheader("🔎 Search for Movies by Genre or Tags")
search_query = st.text_input("Search for a movie or genre:")

if search_query:
    try:
        # **Map Search Queries to Known Genres**
        search_expanded = search_query
        for word, mapped_genre in GENRE_KEYWORDS.items():
            if word in search_query.lower():
                search_expanded = mapped_genre
        
        search_vector = tfidf.transform([search_expanded])  # Fix dimension issue
        similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()

        top_indices = np.argsort(similarity_scores)[-10:][::-1]  # Get top 10 matches
        recommendations = movies_with_tags.iloc[top_indices]

        st.subheader("🎬 Movies Matching Your Search")
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                st.write(f"- **{row['title']}** (Genres: {row['genres'] if pd.notna(row['genres']) else 'No genres listed'})")
        else:
            st.warning("No matching movies found. Try different keywords.")
    except Exception as e:
        st.error(f"An error occurred while searching: {e}")

