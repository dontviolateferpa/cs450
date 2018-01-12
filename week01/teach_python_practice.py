from random import Random
import numpy as np

class movie:
    title = str()
    year = int()
    runtime = int()

    def __init__(self, title, year, runtime):
        self.title = title
        self.year = year
        if (runtime < 0):
            self.runtime = 0
        else:
            self.runtime = runtime

    def __repr__(self):
        return ("" + self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins")

    def runtime_mins_hours(self):
        """return runtime in mins and hours"""
        return self.runtime, self.runtime / 60.0

def create_movie_list():
    """"""
    movie_list = []

    for x in range (0, 5):
        movie_list.append(movie("blah " + str(x), 2000 + x, x * 60))

    return movie_list

def create_list_of_long_movies(movies):
    """"""
    long_movie_list = []

    for movie in movies:
        if movie.runtime > 150:
            long_movie_list.append(movie)
    
    return long_movie_list

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def main():
    """"""
    # create list of movies
    movie_list = create_movie_list()

    # print movie list
    for movie in movie_list:
        print(movie)

    print()

    # make list of long movies
    long_movie_list = create_list_of_long_movies(movie_list)

    for movie in long_movie_list:
        print(movie)
    
    movies_stars_map = {}

    random = Random()

    for movie in movie_list:
        num_stars = random.uniform(0, 5)

        movies_stars_map[movie.title] = num_stars
    
    for movie_title in movies_stars_map:
        num_stars = movies_stars_map[movie_title]
        
        print "{0:.2f}".format(num_stars)
    
    random_movies = get_movie_data()
    
    for movie in random_movies:
        print movie

main()
