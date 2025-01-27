import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.optimizers import Adam
import os

# Load datasets
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')
tags_df = pd.read_csv('data/tags.csv')

# Check if the model exists, otherwise define and train it
model_path = 'models/recommendation_model.h5'

def train_and_save_model(ratings):
    """Train the collaborative filtering model and save it."""
    # Preprocess user and movie mappings
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_id_map = {id_: i for i, id_ in enumerate(user_ids)}
    movie_id_map = {id_: i for i, id_ in enumerate(movie_ids)}
    ratings['user_idx'] = ratings['userId'].map(user_id_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)

    # Define embedding size
    embedding_size = 50

    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    # Embedding layers
    user_embedding = Embedding(input_dim=len(user_ids), output_dim=embedding_size, name='user_embedding')(user_input)
    movie_embedding = Embedding(input_dim=len(movie_ids), output_dim=embedding_size, name='movie_embedding')(movie_input)

    # Flatten embeddings
    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)

    # Dot product for prediction
    dot_product = Dot(axes=1)([user_vector, movie_vector])

    # Compile model
    model = Model(inputs=[user_input, movie_input], outputs=dot_product)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train-test split
    train_data = ratings.sample(frac=0.8, random_state=42)
    test_data = ratings.drop(train_data.index)

    # Train the model
    model.fit(
        [train_data['user_idx'], train_data['movie_idx']],
        train_data['rating'],
        validation_data=(
            [test_data['user_idx'], test_data['movie_idx']],
            test_data['rating']
        ),
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # Save the model
    model.save(model_path)
    return model

# Load or train the model
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.write("Training the model for the first time...")
    model = train_and_save_model(ratings_df)

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
