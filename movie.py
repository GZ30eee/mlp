import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import re  # For regex operations

# Load data
movies_df = pd.read_csv('movies.csv')  # Ensure the correct path to movies.csv
ratings_df = pd.read_csv('ratings.csv')  # Ensure the correct path to ratings.csv

# Preprocess the genres column to ensure there are no null values
movies_df['genres'] = movies_df['genres'].fillna('')

# Step 1: Create a sparse TF-IDF matrix for the genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Step 2: Apply dimensionality reduction (TruncatedSVD)
n_components = min(23, tfidf_matrix.shape[1])  # Set n_components to 23 or lower based on available features
svd = TruncatedSVD(n_components=n_components)
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

# Step 3: Using Approximate Nearest Neighbors (ANN) for similarity calculation
nn_model = NearestNeighbors(metric='cosine', algorithm='auto')
nn_model.fit(reduced_tfidf_matrix)

# Function to clean and normalize the movie title
def normalize_title(title):
    title = title.lower().strip()  # Convert to lowercase and strip leading/trailing whitespaces
    title = re.sub(r'\(\d{4}\)', '', title)  # Remove the year in parentheses (e.g., " (1995)")
    title = title.strip()  # Final strip to remove any extra spaces
    return title

# Step 4: Function to recommend movies by genre
# Step 4: Function to recommend movies by genre
# Step 4: Function to recommend movies by genre
def recommend_movies_by_genre(movie_name, movies_df, n_recommendations=10):
    # Normalize the movie name input
    movie_name = normalize_title(movie_name)
    
    # Check if movie exists in the dataset (case-insensitive and stripped of spaces)
    movie_titles_normalized = movies_df['title'].apply(normalize_title).values

    if movie_name not in movie_titles_normalized:
        return f"Sorry, the movie '{movie_name}' is not in the dataset."
    
    # Get the index of the movie
    movie_idx = movies_df[movies_df['title'].apply(normalize_title) == movie_name].index[0]
    
    # Get the nearest neighbors
    movie_vector = reduced_tfidf_matrix[movie_idx]
    distances, indices = nn_model.kneighbors(movie_vector.reshape(1, -1), n_neighbors=n_recommendations)
    
    recommendations = []
    for i in range(1, len(indices.flatten())):  # Starting from 1 to skip the query movie itself
        recommended_movie = movies_df.iloc[indices.flatten()[i]]
        recommendations.append({
            'title': recommended_movie['title'],
            'genres': recommended_movie['genres'],
            'distance': distances.flatten()[i]
        })
    
    return recommendations

# Streamlit Interface
st.title("Movie Recommendation System")

# Movie selection input
movie_name = st.text_input("Enter a movie name to get recommendations:")

# Check if the movie exists in the dataset
if movie_name:
    recommended_movies = recommend_movies_by_genre(movie_name, movies_df)

    if isinstance(recommended_movies, str):  # If it's an error message
        st.write(recommended_movies)
    else:
        # Display recommendations
        st.write(f"Recommended movies similar to '{movie_name}':")
        for idx, rec in enumerate(recommended_movies, 1):
            st.subheader(f"Recommendation {idx}:")
            st.write(f"**Title:** {rec['title']}")
            st.write(f"**Genres:** {rec['genres']}")
            st.write(f"**Cosine Distance:** {rec['distance']:.4f}")
