🎬 Movie Recommendation System
A machine learning-powered movie recommendation system using Collaborative Filtering, Deep Learning, and Content-Based Filtering. Built with TensorFlow/Keras, Pandas, Scikit-learn, and deployed using Streamlit.

📌 Project Overview
This project aims to build a Movie Recommendation System using user-movie interactions from the MovieLens dataset. The system implements multiple models:

✅ Baseline Model (Top-rated movies)
✅ Collaborative Filtering Model (Matrix Factorization using Embeddings)
✅ Deep Learning Model (Neural Network-based recommendations)
✅ Content-Based Filtering (TF-IDF + Cosine Similarity for genre-based search)
✅ Streamlit Web App for user interaction

📂 Project Structure
📦 movie_recommendation_system
│── 📂 data/ # MovieLens dataset (ratings, movies, tags)
│── 📂 models/ # Saved trained models (recommendation_model.h5)
│── 📜 app.py # Streamlit app for recommendations
│── 📜 train_model.py # Model training script
│── 📜 requirements.txt # Dependencies
│── 📜 README.md # Project documentation
│── 📜 Recommendation_System_Capstone.ipynb # Jupyter Notebook for EDA & Training

🚀 Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

2️⃣ Create a virtual environment & Install dependencies
python -m venv env
source env/bin/activate # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook (For EDA & Model Training)
jupyter notebook
Open Recommendation_System_Capstone.ipynb and execute the cells.

4️⃣ Train the Recommendation Model
python train_model.py
This will train and save the Collaborative Filtering & Deep Learning models.

5️⃣ Run the Streamlit App
streamlit run app.py
The web app will be available at http://localhost:8501/.

📊 Exploratory Data Analysis (EDA)
The MovieLens dataset contains user ratings for movies. We analyzed:

Rating Distribution (Most ratings are around 4.0)
User Activity (Most users rate few movies, some rate thousands)
Movie Popularity (Few movies receive most ratings)
Sparsity of User-Movie Matrix (~99.81% sparse)
📈 Key Visualizations
✅ Rating Distribution (Bar Chart)
✅ Number of Ratings per User (Histogram)
✅ Number of Ratings per Movie (Histogram)

🏗 Feature Engineering
Mapped User IDs & Movie IDs to unique indices
Train-Test Split (80-20%)
Converted timestamps to datetime
Created TF-IDF vectors for Content-Based Filtering
🧠 Models Implemented
1️⃣ Baseline Model (Top-rated movies)
📌 Simply recommends top 10 highest-rated movies.
⚠ Limitation: No personalization for users.

2️⃣ Collaborative Filtering Model (Matrix Factorization)
📌 Uses TensorFlow Embedding Layers to represent users & movies.
📌 Dot Product Layer to compute similarity.
📌 Optimizer: Adam (learning_rate=0.001), Loss Function: MSE

📊 Results:
✔ Test Loss: 0.7701
✔ Test MAE: 0.6679

3️⃣ Deep Learning Model (Neural Network)
📌 Enhances Collaborative Filtering by adding Fully Connected Layers.
📌 Architecture:
✅ Embedding Layers (Users & Movies)
✅ Concatenation Layer
✅ Fully Connected Layers (ReLU Activation, Dropout)
✅ Output Layer (Linear Activation for Rating Prediction)

📊 Results:
✔ Test Loss: 0.6909
✔ Test MAE: 0.6311 (Best performance)

🎭 Content-Based Filtering
Uses TF-IDF Vectorization on movie descriptions/tags.
Cosine Similarity calculates similarity between search queries and movies.
Users can search for movies by entering a genre or keyword.
Example Query: "Sci-Fi"
🔍 Returns Top 10 Sci-Fi movies based on content similarity.

🎬 Deployment with Streamlit
Features:
✅ Search Movies by Genre/Tag (TF-IDF + Cosine Similarity)
✅ Collaborative & Deep Learning Recommendations
✅ Interactive UI for User Input
✅ Top Movie Recommendations Displayed
