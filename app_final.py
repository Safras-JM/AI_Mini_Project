import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import nltk
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
from sklearn.metrics.pairwise import cosine_similarity as cs

# Load the preprocessed dataset and distance similarity matrix
music = pickle.load(open('first_20000_songs_dataset.pkl', 'rb'))
similarity = pickle.load(open('distance_similarity.pkl', 'rb'))

# Initialize the Spotify client
CLIENT_ID = "86262ff461104aeca7510711c6f86cf3"
CLIENT_SECRET = "a1f1ea00624c44318d5e418a95eb3631"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def preprocess_lyrics(lyrics, stemmer):
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^a-z0-9\s]', ' ', lyrics)
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'\r', ' ', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics)
    lyrics = lyrics.strip()
    return tokenization(lyrics, stemmer)

def tokenization(txt, stemmer):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

def get_similarity_with_input(lyrics, vector, matrix, stemmer):
    preprocessed_lyrics = preprocess_lyrics(lyrics, stemmer)
    input_vector = vector.transform([preprocessed_lyrics])
    similarity = cs(input_vector, matrix).flatten()
    return similarity

def recommend_based_on_lyrics(input_lyrics):
    input_similarity = get_similarity_with_input(input_lyrics, vector, matrix, stemmer)
    distances = sorted(enumerate(input_similarity), reverse=True, key=lambda x: x[1])

    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)

    return recommended_music_names, recommended_music_posters

# Tokenization and stemming initialization
stemmer = LancasterStemmer()

# Vectorization initialization
vector = tfv(analyzer='word', stop_words='english')
matrix = vector.fit_transform(music['text'])

# Streamlit web app code
st.header('Music Recommender System')

input_lyrics = st.text_area("Enter the lyrics for music recommendation:")

if st.button('Show Recommendation') and input_lyrics:
    recommended_music_names, recommended_music_posters = recommend_based_on_lyrics(input_lyrics)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
