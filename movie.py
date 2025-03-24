import streamlit as st
import pandas as pd
import random

# Load the dataset
movies = pd.read_csv("movies.csv")

# Function to get recommendations based on genre
def get_movies_by_genre(selected_genre):
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    return filtered_movies.sample(min(5, len(filtered_movies)))

# Function to find similar movies by title
def get_similar_movies(movie_title):
    movie = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
    if not movie.empty:
        genres = movie.iloc[0]['genres'].split('|')
        related_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in genres))]
        return related_movies.sample(min(5, len(related_movies)))
    return pd.DataFrame()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.sidebar.header("Choose an Option")

# Search by Genre
st.sidebar.subheader("Find Movies by Genre")
genres_list = set("|".join(movies['genres'].dropna()).split('|'))
selected_genre = st.sidebar.selectbox("Select a genre", ["Select"] + sorted(genres_list))

if selected_genre != "Select":
    st.subheader(f"Recommended {selected_genre} Movies:")
    st.table(get_movies_by_genre(selected_genre))

# Search by Movie Title
st.sidebar.subheader("Find Similar Movies")
movie_search = st.sidebar.text_input("Enter movie title")
if movie_search:
    st.subheader(f"Movies Similar to '{movie_search}':")
    results = get_similar_movies(movie_search)
    if not results.empty:
        st.table(results)
    else:
        st.write("No similar movies found.")

# Random Movie Suggestion
st.sidebar.subheader("Surprise Me!")
if st.sidebar.button("Get a Random Movie"):
    random_movie = movies.sample(1)
    st.subheader("Here's a Movie for You:")
    st.write(random_movie[['title', 'genres']])
