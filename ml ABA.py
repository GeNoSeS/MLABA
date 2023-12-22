#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[27]:


data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')
artist_data = pd.read_csv('data_by_artist.csv')


# In[34]:


data.sample(5)


# In[35]:


genre_data.sample(5)


# In[36]:


year_data.sample(5)


# In[37]:


artist_data.sample(5)


# In[38]:


datasets = [("data", data), ("genre_data", genre_data), ("year_data", year_data), ("artist_data", artist_data)]


# In[46]:


data['year'] = pd.to_datetime(data['year'], format='%Y')
data['release_date'] = pd.to_datetime(data['release_date'],format="mixed")
year_data['year'] = pd.to_datetime(year_data['year'], format='%Y')


# In[47]:


for name, df in datasets:
    # print some info about the datasets
    print(f"Info about the dataset: {name}")
    print("-"*30)
    print(df.info())
    print()


# In[48]:


for name, df in datasets:
    # Check for missing values in the datasets
    print(f"Missing Values in: {name}")
    print("-"*30)
    print(df.isnull().sum())
    print()


# In[49]:


for name, df in datasets:
    # Check the unique values in the dataset
    print(f"Unique Values in: {name}")
    print("-"*30)
    print(df.nunique())
    print()


# In[50]:


# Popularity Trends Over Years
fig = px.line(year_data, x='year', y='popularity', title='Popularity Trends Over Years')
fig.show()


# In[51]:


# Convert release_date to datetime and extract decade
data['release_decade'] = (data['release_date'].dt.year // 10) * 10

# Count the number of songs per decade
decade_counts = data['release_decade'].value_counts().sort_index()

# Create a bar chart for songs per decade
fig = px.bar(x=decade_counts.index, y=decade_counts.values, labels={'x': 'Decade', 'y': 'Number of Songs'},
             title='Number of Songs per Decade')
fig.update_layout(xaxis_type='category')
fig.show()


# In[52]:


fig = px.scatter(year_data, x='year', y='tempo', color='tempo', size='popularity',
                 title='Tempo Changes Over Years', labels={'tempo': 'Tempo'})
fig.show()


# In[53]:


# Average Danceability Over Years
fig = px.line(year_data, x='year', y='danceability', title='Average Danceability Over Years')
fig.show()


# In[54]:


# Danceability and Energy Over Years
fig = go.Figure()

fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['danceability'], mode='lines', name='Danceability'))
fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['energy'], mode='lines', name='Energy'))

fig.update_layout(title='Danceability and Energy Over Years', xaxis_title='Year', yaxis_title='Value')
fig.show()


# In[55]:


# Energy and Acousticness Over Years
fig = go.Figure()

fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['energy'], mode='lines', name='Energy'))
fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['acousticness'], mode='lines', name='Acousticness'))

fig.update_layout(title='Energy and Acousticness Over Years', xaxis_title='Year', yaxis_title='Value')
fig.show()


# In[56]:


# Speechiness and Instrumentalness Over Years
fig = go.Figure()

fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['speechiness'], mode='lines', name='Speechiness'))
fig.add_trace(go.Scatter(x=year_data['year'], y=year_data['instrumentalness'], mode='lines', name='Instrumentalness'))

fig.update_layout(title='Speechiness and Instrumentalness Over Years', xaxis_title='Year', yaxis_title='Value')
fig.show()


# In[57]:


# Valence Distribution by Release Year
fig = px.box(data, x=data['release_date'].dt.year, y='valence', title='Valence Distribution by Release Year')
fig.show()


# In[58]:


# Release Frequency Over Years
release_counts = data['release_date'].dt.year.value_counts().reset_index()
release_counts.columns = ['Year', 'Count']

fig = px.bar(release_counts, x='Year', y='Count', title='Release Frequency Over Years')
fig.show()


# In[59]:


# Genre Analysis: Top Genres by Popularity
top_10_genre_data = genre_data.nlargest(10, 'popularity')

fig = px.bar(top_10_genre_data, x='popularity', y='genres', orientation='h',
             title='Top Genres by Popularity', color='genres')
fig.show()


# In[60]:


fig = px.bar(top_10_genre_data, x='genres', y='danceability', color='genres',
             title='Danceability Distribution for Top 10 Popular Genres')
fig.show()


# In[61]:


fig = px.bar(top_10_genre_data, x='genres', y='energy', color='genres',
             title='Energy Distribution for Top 10 Popular Genres')
fig.show()


# In[62]:


fig = px.bar(top_10_genre_data, x='genres', y='valence', color='genres',
             title='Valence Distribution for Top 10 Popular Genres')
fig.show()


# In[63]:


fig = px.bar(top_10_genre_data, x='genres', y='acousticness', color='genres',
             title='Acousticness Distribution for Top 10 Popular Genres')
fig.show()


# In[64]:


fig = px.bar(top_10_genre_data, x='genres', y='instrumentalness', color='genres',
             title='Instrumentalness Distribution for Top 10 Popular Genres')
fig.show()


# In[65]:


top_10_artist_data = artist_data.nlargest(10, 'popularity')

fig = px.bar(top_10_artist_data, x='popularity', y='artists', orientation='h', color='artists',
             title='Top Artists by Popularity')
fig.show()


# In[66]:


fig = px.scatter(top_10_artist_data, x='speechiness', y='instrumentalness', color='artists',
                 size='popularity', hover_name='artists',
                 title='Speechiness vs. Instrumentalness for Top Artists')
fig.show()


# In[67]:


fig = px.scatter(top_10_artist_data, x='danceability', y='energy', color='artists',
                 size='popularity', hover_name='artists',
                 title='Danceability vs. Energy for Top 10 Popular Artists')
fig.show()


# In[68]:


top_songs = data.nlargest(10, 'popularity')

fig = px.bar(top_songs, x='popularity', y='name', orientation='h',
             title='Top Songs by Popularity', color='name')
fig.show()


# In[69]:


fig = px.scatter(top_songs, x='danceability', y='energy', color='popularity',
                 size='popularity', hover_name='name',
                 title='Danceability vs. Energy for Top Songs')
fig.show()


# In[70]:


fig = px.scatter(top_songs, x='speechiness', y='instrumentalness', color='popularity',
                 size='popularity', hover_name='name',
                 title='Speechiness vs. Instrumentalness for Top Songs')
fig.show()


# In[71]:


data['year'] = data['year'].dt.year


# In[72]:


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'year',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


# In[73]:


def get_song_data(name, data):
    try:
        return data[data['name'].str.lower() == name].iloc[0]
        return song_data
    except IndexError:
        return None


# In[74]:


def get_mean_vector(song_list, data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], data)
        if song_data is None:
            print('Warning: {} does not exist in the dataset'.format(song['name']))
            return None
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


# In[75]:


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


# In[76]:


min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data[number_cols])

# Standardize the normalized data using Standard Scaler
standard_scaler = StandardScaler()
scaled_normalized_data = standard_scaler.fit_transform(normalized_data)


# In[77]:


def recommend_songs(seed_songs, data, n_recommendations=10):
    metadata_cols = ['name', 'artists', 'year']
    song_center = get_mean_vector(seed_songs, data)
    
    # Return an empty list if song_center is missing
    if song_center is None:
        return []
    
    # Normalize the song center
    normalized_song_center = min_max_scaler.transform([song_center])
    
    # Standardize the normalized song center
    scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)
    
    # Calculate Euclidean distances and get recommendations
    distances = cdist(scaled_normalized_song_center, scaled_normalized_data, 'euclidean')
    index = np.argsort(distances)[0]
    
    # Filter out seed songs and duplicates, then get the top n_recommendations
    rec_songs = []
    for i in index:
        song_name = data.iloc[i]['name']
        if song_name not in [song['name'] for song in seed_songs] and song_name not in [song['name'] for song in rec_songs]:
            rec_songs.append(data.iloc[i])
            if len(rec_songs) == n_recommendations:
                break
    
    return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')


# In[78]:


seed_songs = [
    {'name': 'Paranoid'},
    {'name': 'Blinding Lights'},
    # Add more seed songs as needed
]
seed_songs = [{'name': name['name'].lower()} for name in seed_songs]

# Number of recommended songs
n_recommendations = 15

# Call the recommend_songs function
recommended_songs = recommend_songs(seed_songs, data, n_recommendations)

# Convert the recommended songs to a DataFrame
recommended_df = pd.DataFrame(recommended_songs)

# Print the recommended songs
for idx, song in enumerate(recommended_songs, start=1):
    print(f"{idx}. {song['name']} by {song['artists']} ({song['year']})")


# In[ ]:




