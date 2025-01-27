import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load datasets
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')
tags_df = pd.read_csv('data/tags.csv')

# Check for the model and load it
model_path = 'models/recommendation_model.h5'
try:
    # Load the model with the correct loss function mapping
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Precompute TF-IDF matrix for movie tags
tfidf = TfidfVectorizer(stop_words='english')
tags_df['tag'] = tags_df['tag'].fillna('')  # Handle missing tags
movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = pd.merge(movies_df, movie_tags, on='movieId', how='left')
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['tag'])

# Streamlit App
st.title("Enhanced Movie Recommendation System")
st.write("Enter your user ID or search for movies using natural language.")

# User Feedback and Sentiment Analysis
st.subheader("User Feedback")
feedback = st.text_area("Tell us about the type of movies you like (e.g., 'I enjoy action-packed thrillers').")
temp_user_id = None  # Temporary user ID
temp_ratings = []

if feedback:
    sentiment = TextBlob(feedback).sentiment
    st.write(f"Sentiment Analysis: Polarity = {sentiment.polarity:.2f}, Subjectivity = {sentiment.subjectivity:.2f}")

    # Generate a temporary user ID
    temp_user_id = max(ratings_df['userId']) + 1
    st.write(f"Generated User ID for Recommendations: {temp_user_id}")

    # Extract preferences (genres or tags) from feedback
    preferred_genres = []
    for genre in movies_df['genres'].unique():
        if genre.lower() in feedback.lower():
            preferred_genres.append(genre)

    st.write("Extracted Preferences (Genres):", preferred_genres)

    # Simulate ratings for the temporary user based on preferences
    for _, row in movies_df.iterrows():
        if any(genre in row['genres'] for genre in preferred_genres):
            rating = 4.0 + (sentiment.polarity * 1.0)  # Boost ratings for positive sentiment
            temp_ratings.append({'userId': temp_user_id, 'movieId': row['movieId'], 'rating': min(max(rating, 1.0), 5.0)})
        else:
            rating = 2.0 + (sentiment.polarity * 0.5)  # Lower ratings but still influenced by sentiment
            temp_ratings.append({'userId': temp_user_id, 'movieId': row['movieId'], 'rating': min(max(rating, 1.0), 5.0)})

    st.write("Temporary ratings generated for collaborative filtering.")
    st.info("Use the generated User ID for personalized recommendations.")

# Search Functionality
st.subheader("Search for Movies")
search_query = st.text_input("Search for a movie or genre:")
if search_query:
    # Compute similarity
    search_vector = tfidf.transform([search_query])
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:10]  # Top 10 matches
    recommendations = movies_with_tags.iloc[top_indices][['title', 'genres', 'tag']]

    # Display results
    st.write("Movies matching your search:")
    for _, row in recommendations.iterrows():
        st.write(f"- **{row['title']}** (Genres: {row['genres']})")
        st.caption(f"Tags: {row['tag']}")

# Recommendation by User ID
st.subheader("Recommendations by User ID")
user_input = st.text_input("User ID", "")
if user_input or temp_user_id:
    try:
        user_id = int(user_input) if user_input else temp_user_id
        user_ids = ratings_df['userId'].unique()

        # Add temporary user ratings to the dataset
        if temp_user_id and temp_ratings:
            temp_df = pd.DataFrame(temp_ratings)
            ratings_with_temp = pd.concat([ratings_df, temp_df])
        else:
            ratings_with_temp = ratings_df

        # Generate collaborative filtering recommendations
        if user_id in user_ids or user_id == temp_user_id:
            user_idx = user_id - 1  # Adjust for zero-based index in the model
            all_movie_indices = np.array(list(range(len(movies_df))))
            user_array = np.array([user_idx] * len(all_movie_indices))

            # Predict ratings
            predictions = model.predict([user_array, all_movie_indices], verbose=0).flatten()
            top_indices = predictions.argsort()[::-1][:10]
            recommended_movies = movies_df.iloc[top_indices]

            st.write(f"Top recommendations for User ID {user_id}:")
            for _, row in recommended_movies.iterrows():
                st.write(f"- **{row['title']}** (Genres: {row['genres']})")
        else:
            st.error("User ID not found. Please try another.")
    except ValueError:
        st.error("Invalid User ID. Please enter a numeric value.")
