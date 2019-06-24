from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
import pandas as pd
import numpy as np
import webbrowser
# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        moviname=request.args.get('search')
        file=open("./pyscripts/inputMovie.txt","r+")
        file.truncate(0)
        file.write(str(moviname))
        file.close()
    # Reading ratings file
# Ignore the timestamp column
        ratings = pd.read_csv('./pyscripts/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
        users = pd.read_csv('./pyscripts/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
        movies = pd.read_csv('./pyscripts/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
# Join all 3 files into one dataframe
        dataset = pd.merge(pd.merge(movies, ratings),users)
# Display 20 movies with highest ratings
        dataset[['title','genres','rating']].sort_values('rating', ascending=False)
# Make a census of the genre keywords
        genre_labels = set()
        for s in movies['genres'].str.split('|').values:
            genre_labels = genre_labels.union(set(s))

# Function that counts the number of times each of the genre keywords appear
        def count_word(dataset, ref_col, census):
            keyword_count = dict()
            for s in census: 
                keyword_count[s] = 0
            for census_keywords in dataset[ref_col].str.split('|'):        
                if type(census_keywords) == float and pd.isnull(census_keywords): 
                    continue        
                for s in [s for s in census_keywords if s in census]: 
                    if pd.notnull(s): 
                        keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords by frequency
            keyword_occurences = []
            for k,v in keyword_count.items():
                keyword_occurences.append([k,v])
            keyword_occurences.sort(key = lambda x:x[1], reverse = True)
            return keyword_occurences, keyword_count

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
        keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
# Break up the big genre string into a string array
        movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
        movies['genres'] = movies['genres'].fillna("").astype('str')
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(movies['genres'])
        from sklearn.metrics.pairwise import linear_kernel
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Build a 1-dimensional array with movie titles
        titles = movies['title']
        indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
        def genre_recommendations(title):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            movie_indices = [i[0] for i in sim_scores]
            return titles.iloc[movie_indices]
        fil=open("./pyscripts/inputMovie.txt","r+")
        movi=fil.readline()
        fil.close()
        fil1=open("./pyscripts/outputMovie.txt","r+")
        fil1.write(str(genre_recommendations(movi)))
        fil1.close()
        webbrowser.open('file:///C:/Users/Vaibhav/Desktop/recommendersys/templates/page2.html')
        return "1"
    elif request.method == 'POST':
        ui=request.form['uid']
        slid=request.form['slider']
        fille=open("./pyscripts/inputDet.txt","r+")
        fille.truncate(0)
        fille.write(ui)
        fille.write("\n")
        fille.write(slid)
        fille.close()
        # Reading ratings file
        ratings = pd.read_csv('./pyscripts/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating', 'timestamp'])

# Reading users file
        users = pd.read_csv('./pyscripts/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
        movies = pd.read_csv('./pyscripts/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
        n_users = ratings.user_id.unique().shape[0]
        n_movies = ratings.movie_id.unique().shape[0]

        Ratings = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
        R = Ratings.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
        sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)

        from scipy.sparse.linalg import svds
        U, sigma, Vt = svds(Ratings_demeaned, k = 50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)
        def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
            user_row_number = userID - 1 # User ID starts at 1, not 0
            sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the movie information.
            user_data = original_ratings[original_ratings.user_id == (userID)]
            user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['rating'], ascending=False)
                 )

    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
            recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
                merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                    left_on = 'movie_id',
                    right_on = 'movie_id').
                rename(columns = {user_row_number: 'Predictions'}).
                sort_values('Predictions', ascending = False).
                                iloc[:num_recommendations, :-1]
                            )

            return user_full, recommendations
        fil=open("./pyscripts/inputDet.txt","r")

        uidd=int(fil.readline())

        numm=int(fil.readline())

        fil.close()
        already_rated, predictions = recommend_movies(preds, uidd, movies, ratings, numm)
#print(already_rated)
        fil1=open("./pyscripts/outputDet.txt","w")
        fil1.write(str(predictions))
        fil1.close()
        webbrowser.open('file:///C:/Users/Vaibhav/Desktop/recommendersys/templates/page3.html')
        return "1"

        
    
    

    
if __name__ == '__main__':
   app.run(debug = False)    