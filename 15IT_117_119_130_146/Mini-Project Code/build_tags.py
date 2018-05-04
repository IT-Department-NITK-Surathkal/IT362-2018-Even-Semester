import json
import pandas as pd
import random
from collections import defaultdict
from math import log

def get_movie(movie_id):
    tags = pd.read_csv("~/Projects/IR/Movie-Recommender-System/dataset/movies_small/tags.csv")


    if movie_id not in list(tags['movieId']):
        return []
    tags = tags[tags['movieId'] == movie_id]
    tags = list(tags['tag'])
    tags = [x.split(',') for x in tags]
    tag = []
    for ts in tags:
        for t in ts:
            if t.lower() not in tag:
                tag.append(t.lower())

    return tag



def add_tags():
    movies = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/movies.csv')
    tags = []
    for id, row in movies.iterrows():
        print(row['movieId'])
        user_tags = get_movie(row['movieId'])
        genres = str(row['genres'])
        genres = genres.lower().split('|')
        temp = user_tags + genres
        tags.append('_'.join(temp))
    movies['tags'] = tags
    movies.to_csv('./Movie-Recommender-System/dataset/movies_small/modified_movies.csv')


def create_coview():
    movies_t = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/modified_movies.csv')
    movies = movies_t[:100]
    total_coviews = []
    movieIds = list(movies['movieId'])
    for id, row in movies.iterrows():
        tags = row['genres'].split('|')
        coviews = []
        for tag in tags:
            for id2, row2 in movies.iterrows():
                if id != id2 and tag in row2['genres'].split('|'):
                    if row2['movieId'] not in coviews:
                        coviews.append(row2['movieId'])
        if len(coviews) > 10:
            random.shuffle(coviews)
            coviews = coviews[:10]
        num_left = 10 - len(coviews)
        movieCopy = movieIds[:]
        random.shuffle(movieCopy)
        coviews = coviews + movieCopy[:num_left]
        total_coviews.append(json.dumps(coviews))

    movies['coviews'] = total_coviews
    movies.to_csv('./Movie-Recommender-System/dataset/movies_small/coview_movies.csv')


def c(tao, movieId):
    movies = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/coview_movies.csv')
    row = movies[movies['movieId'] == movieId]
    row = row.iloc[0].squeeze()

    coviews = json.loads(row['coviews'])
    counter = 0
    for id in coviews:
        row2 = movies[movies['movieId'] == id]
        row2 = row2.iloc[0].squeeze()
        if tao.lower() in row2['genres'].lower().split('|'):
            counter += 1

    return counter


def genreFreq():
    movies = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/coview_movies.csv')
    genres = []
    for i, r in movies.iterrows():
        g = r['genres'].lower().split('|')
        for genre in g:
            if genre not in genres:
                genres.append(genre)

    freq = defaultdict(int)
    for i, r in movies.iterrows():
        g = r['genres'].lower().split('|')
        for genre in g:
            freq[genre] += 1

    return dict(freq)


def get_score(movieId1, movieId2):
    movies = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/coview_movies.csv')
    row1 = movies[movies['movieId'] == movieId1]
    row1 = row1.iloc[0].squeeze()
    genres1 = row1['genres'].lower().split('|')

    row2 = movies[movies['movieId'] == movieId2]
    row2 = row2.iloc[0].squeeze()
    genres2 = row2['genres'].lower().split('|')

    common = []
    for g in genres1:
        if g in genres2:
            common.append(g)

    df = genreFreq()
    score = 0
    for g in common:
        score += (c(g, movieId1)*c(g, movieId2))/log(1 + df[g], 2)

    return score


def suggestionsList(movieId):
    movies = pd.read_csv('./Movie-Recommender-System/dataset/movies_small/coview_movies.csv')
    suggestions = []
    for i, r in movies.iterrows():
        id_ = int(r['movieId'])
        if id_ != movieId:
            score = get_score(movieId, id_)
            suggestions.append([id_, r['title'], score])
        else:
            movie_name = r['title']

    suggestions = sorted(suggestions, key=lambda a: a[2])
    suggestions.reverse()
    suggestions = suggestions[:-(len(suggestions)-10)]
    return movie_name, suggestions
