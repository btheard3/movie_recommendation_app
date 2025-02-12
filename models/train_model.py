import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import os

# Load dataset
ratings_df = pd.read_csv("data/ratings.csv")

# Create user and movie mappings
user_ids = ratings_df["userId"].unique()
movie_ids = ratings_df["movieId"].unique()

user_id_map = {id: i for i, id in enumerate(user_ids)}
movie_id_map = {id: i for i, id in enumerate(movie_ids)}

ratings_df["user_idx"] = ratings_df["userId"].map(user_id_map).astype(np.int32)  # ✅ Ensure int32
ratings_df["movie_idx"] = ratings_df["movieId"].map(movie_id_map).astype(np.int32)  # ✅ Ensure int32

num_users = len(user_ids)
num_movies = len(movie_ids)

# Define the Collaborative Filtering Model
model = Sequential([
    Embedding(input_dim=num_users, output_dim=50, name="user_embedding"),  # ✅ Removed input_length=1
    Flatten(),
    Dense(50, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Convert input arrays correctly
X_train = np.array(ratings_df["user_idx"].values)  # ✅ Convert to NumPy array
y_train = np.array(ratings_df["rating"].values)

# Train the model
model.fit(
    x=X_train,  # ✅ Fixed issue: Pass as NumPy array
    y=y_train,
    batch_size=64,
    epochs=3,
    verbose=1
)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/recommendation_model.h5")

print("✅ Model training complete. Saved as recommendation_model.h5")
